"""On-demand session-v2 viewer. Memmaps, images and meshes are computed on request."""
import argparse
import functools
import hashlib
import json
import math
import os
import re
from collections import OrderedDict
import sys
import threading
from pathlib import Path
import urllib.request
import urllib.parse
import urllib.error

import numpy as np
import cv2
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response, HTMLResponse
import uvicorn
import orjson

HERE=Path(__file__).parent
REPO=HERE.parents[2]
sys.path.insert(0, str(REPO))
from scripts.data.session.hand_crops import active as active_crop, crop_record, crop_bytes
from egoemg.data_processing.calibration_io import read_json as read_calibration_json, load_calibration

@functools.lru_cache(maxsize=256)
def cached_json(path,mtime,size):return read_calibration_json(path)

def read(path):
    path=Path(path);stat=path.stat()
    return cached_json(str(path),stat.st_mtime_ns,stat.st_size)

def fast_json(value):
    def default(item):
        if isinstance(item,np.ndarray):return np.ascontiguousarray(item).view(np.ndarray)
        if isinstance(item,np.generic):return item.item()
        raise TypeError(type(item).__name__)
    return Response(orjson.dumps(value,option=orjson.OPT_SERIALIZE_NUMPY,default=default),media_type='application/json')

app=FastAPI(); SOURCES={}; SESSIONS={}; CAPS={}; LOCK=threading.RLock(); CAP_LOCK=threading.RLock(); COLLECTION=None
cv2.setNumThreads(1)

def remote(url,path,query,timeout=30,with_headers=False):
    request=url.rstrip('/')+path+'?'+urllib.parse.urlencode(query)
    try:
        with urllib.request.urlopen(request,timeout=timeout) as r:
            payload=r.read();mime=r.headers.get_content_type()
            return (payload,mime,{k:v for k,v in r.headers.items() if k.lower().startswith('x-') or k.lower()=='cache-control'}) if with_headers else (payload,mime)
    except urllib.error.HTTPError as exc:
        detail=exc.read().decode('utf-8',errors='replace')
        try:detail=json.loads(detail).get('detail',detail)
        except ValueError:pass
        raise HTTPException(exc.code,detail) from exc

def clean(value):
    if isinstance(value,np.ndarray):return clean(value.tolist())
    if isinstance(value,np.generic):return clean(value.item())
    if isinstance(value,float) and not math.isfinite(value):return None
    if isinstance(value,dict):return {k:clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(v) for v in value]
    return value

@functools.lru_cache(maxsize=512)
def mapped(path,dtype,shape):
    if not np.prod(shape):return np.empty(shape,dtype=dtype)
    return np.memmap(path,dtype=dtype,mode='r',shape=shape)

def data(folder,spec):return mapped(str(folder/spec['path']),spec['dtype'],tuple(spec['shape']))

def row_at(times,t):
    return int(np.clip(np.searchsorted(times,t,side='right')-1,0,len(times)-1))

@functools.lru_cache(maxsize=256)
def ordered(path,dtype,shape):
    ts=mapped(path,dtype,shape)
    monotone=all(np.all(np.diff(ts[max(0,i-1):min(i+65536,len(ts))])>=0) for i in range(0,len(ts),65536))
    if monotone:return ts,None
    indices=np.argsort(ts,kind='stable')
    return ts[indices],indices

def time_axis(folder,spec):return ordered(str(folder/spec['path']),spec['dtype'],tuple(spec['shape']))

def sample_row(folder,h,t):
    ts,order=time_axis(folder,h['arrays']['timestamps']);row=row_at(ts,t)
    return int(order[row]) if order is not None else row

def resolve(key):
    if key not in SESSIONS:raise HTTPException(404,'Unknown session')
    source,entry=SESSIONS[key]
    if source.get('url'):return source,entry,None,None
    path=Path(source['path'])/entry['path']
    return source,entry,path.parent,read(path)

def span(folder,s):
    axes=[time_axis(folder,h['arrays']['timestamps'])[0] for h in s['hands'].values() if h]
    return min(int(a[0]) for a in axes),max(int(a[-1]) for a in axes)

@functools.lru_cache(maxsize=128)
def camera_axis(path,dtype,shape,mask_path,mask_dtype):
    ts=mapped(path,dtype,shape)
    mask=mapped(mask_path,mask_dtype,shape) if mask_path else ts>0
    valid=np.flatnonzero(mask)
    if np.any(np.diff(ts[valid])<0):valid=valid[np.argsort(ts[valid],kind='stable')]
    return ts[valid],valid

def continuity_valid(folder,arrays,row,t):
    spec=arrays.get('continuity_segment')
    if not spec:return True
    intervals=read(folder/'session.json').get('timeline_compaction',{}).get('intervals',[])
    if not intervals:return True
    segment=np.searchsorted([item['output_start_us'] for item in intervals],t,side='right')-1
    return segment>=0 and t<intervals[segment]['output_end_us'] and int(data(folder,spec)[row])==segment

def camera_row(folder,c,t):
    spec=c['arrays']['timestamps'];mask=c['arrays'].get('timestamp_valid')
    ts,valid=camera_axis(str(folder/spec['path']),spec['dtype'],tuple(spec['shape']),str(folder/mask['path']) if mask else None,mask['dtype'] if mask else None)
    if not len(valid):return None
    idx=valid[row_at(ts,t)]
    if abs(int(data(folder, spec)[idx]) - t) > c.get('maximum_frame_age_us', 100000):
        return None
    quality = c['arrays'].get('frame_valid')
    if quality and not data(folder, quality)[idx]:
        return None
    if not continuity_valid(folder,c['arrays'],idx,t):return None
    return int(idx)

def first_camera_time(folder,c,session_start):
    """Return the first usable camera time inside the session's hand timeline."""
    arrays=c['arrays'];spec=arrays['timestamps'];mask=arrays.get('timestamp_valid')
    ts,rows=camera_axis(str(folder/spec['path']),spec['dtype'],tuple(spec['shape']),
                        str(folder/mask['path']) if mask else None,
                        mask['dtype'] if mask else None)
    for pos in range(int(np.searchsorted(ts,session_start)),len(rows)):
        timestamp=int(ts[pos])
        if camera_row(folder,c,timestamp) is not None:
            return (timestamp-session_start)/1e6
    return None

class VideoReader:
    def __init__(self,path):
        self.path=path;self.lock=threading.RLock();self.cap=None;self.index=-1;self.image=None;self.jpeg=None
        self.container=None;self.iterator=None;self.pts=None
        index_path=Path(path).with_suffix('.frame_pts.npy')
        if index_path.exists():self.pts=np.load(index_path,allow_pickle=False)
    def close(self):
        with self.lock:
            if self.cap is not None:self.cap.release()
            if self.container is not None:self.container.close()
            self.container=None;self.iterator=None
            self.cap=None;self.index=-1;self.image=None;self.jpeg=None
    def decode(self,index):
        if self.pts is not None:return self.decode_exact(index)
        if self.cap is None:
            params=[cv2.CAP_PROP_N_THREADS,2] if hasattr(cv2,'CAP_PROP_N_THREADS') else []
            self.cap=cv2.VideoCapture(self.path,cv2.CAP_FFMPEG,params)
            if not self.cap.isOpened():self.cap=cv2.VideoCapture(self.path)
        if index==self.index and self.image is not None:return self.image
        distance=index-self.index
        if self.image is not None and 0<distance<=32:
            for _ in range(distance-1):
                if not self.cap.grab():raise HTTPException(404,'Video frame cannot decode')
        else:self.cap.set(cv2.CAP_PROP_POS_FRAMES,index)
        ok,image=self.cap.read()
        if not ok:raise HTTPException(404,'Video frame cannot decode')
        self.image=image;self.index=index;self.jpeg=None
        return image
    def decode_exact(self,index):
        """Seek by packet PTS, then verify the physical display ordinal for VFR."""
        if index==self.index and self.image is not None:return self.image
        if index<0 or index>=len(self.pts):raise HTTPException(404,'Video frame outside PTS index')
        if self.container is None:
            import av
            self.container=av.open(self.path);self.stream=self.container.streams.video[0]
            self.stream.codec_context.thread_count=2
        target=int(self.pts[index]);distance=index-self.index
        if self.iterator is None or not 0<distance<=32:
            self.container.seek(target,stream=self.stream,backward=True,any_frame=False)
            self.iterator=iter(self.container.decode(self.stream))
        for frame in self.iterator:
            if frame.pts is None or frame.pts<target:continue
            if frame.pts!=target:raise HTTPException(404,'Video PTS index does not match decoded frame')
            self.image=frame.to_ndarray(format='bgr24');self.index=index;self.jpeg=None
            return self.image
        raise HTTPException(404,'Video frame cannot decode at indexed PTS')

def video_reader(path):
    path=Path(path)
    video_stat=path.stat()
    pts_path=path.with_suffix('.frame_pts.npy')
    pts_stat=pts_path.stat() if pts_path.exists() else None
    key=(str(path),video_stat.st_dev,video_stat.st_ino,video_stat.st_size,
         video_stat.st_mtime_ns,pts_stat.st_ino if pts_stat else None,
         pts_stat.st_mtime_ns if pts_stat else None)
    evicted=[]
    with CAP_LOCK:
        for old_key in list(CAPS):
            if old_key[0]==str(path) and old_key!=key:evicted.append(CAPS.pop(old_key))
        if key not in CAPS:CAPS[key]=VideoReader(str(path))
        reader=CAPS.pop(key);CAPS[key]=reader
        while len(CAPS)>8:evicted.append(CAPS.pop(next(iter(CAPS))))
    for old in evicted:old.close()
    return reader

class Geometry:
    def __init__(self):
        # The archived MANO pickle uses chumpy's NumPy-1 scalar aliases.
        if 'float_' not in np.__dict__:np.float_=np.float64
        if 'complex_' not in np.__dict__:np.complex_=np.complex128
        import torch
        self.torch=torch;torch.set_num_threads(2)
        from manotorch.manolayer import ManoLayer
        asset_root=Path(os.environ.get('EGOEMG_ROOT',str(REPO)))
        assets=Path(os.environ.get('MANO_ASSETS_ROOT',str(asset_root/'pretrained_models/mano_assets')))
        if not (assets/'models/MANO_RIGHT.pkl').is_file():
            raise FileNotFoundError('MANO mesh rendering needs separately obtained models/MANO_RIGHT.pkl; set MANO_ASSETS_ROOT or install under $EGOEMG_ROOT/pretrained_models/mano_assets')
        self.layer=ManoLayer(use_pca=False,mano_assets_root=str(assets),flat_hand_mean=False).eval()
        self.faces=self.layer.th_faces.cpu().numpy().tolist()
        from egoemg.kinematics import load_default_hand_model, broadcast_hand_model_to
        from egoemg.UmeTrack.lib.common.hand_skinning import skin_landmarks, _skin_points
        self.hand=broadcast_hand_model_to(load_default_hand_model(),(1,));self.skin=skin_landmarks
        self.skin_points=_skin_points
        self.umetrack_faces=self.hand.mesh_triangles[0].long().cpu().numpy().tolist()
        self.cache=OrderedDict()
    def fit(self,x,y):
        xc=x.mean(0);yc=y.mean(0);u,_,vt=np.linalg.svd((x-xc).T@(y-yc));r=vt.T@u.T
        if np.linalg.det(r)<0:vt[-1]*=-1;r=vt.T@u.T
        return r,yc-r@xc
    def render(self,hand,h,folder,row):
        key=(str(folder),hand,int(row),h['arrays'].get('joint_angles',{}).get('path'),h['arrays'].get('umetrack_world_transform',{}).get('path'))
        if key in self.cache:
            value=self.cache.pop(key);self.cache[key]=value;return value
        value=self._render(hand,h,folder,row);self.cache[key]=value
        while len(self.cache)>128:self.cache.popitem(last=False)
        return value
    def _render(self,hand,h,folder,row):
        spec=h['arrays'].get('mano_pose')
        if not spec:return dict(status='MANO labels are not available')
        valid_spec=h['arrays'].get('mano_valid') or h['arrays'].get('joint_angles_valid') or h['arrays'].get('label_valid')
        mano_valid=not valid_spec or bool(data(folder,valid_spec)[row])
        beta_spec=h['arrays'].get('mano_beta')
        if not beta_spec:return dict(status='MANO shape parameters are missing')
        beta_index=0
        if beta_spec['shape'][0]==h['rows']:beta_index=row
        else:
            for record in h.get('recordings',[]):
                if record['samples'][0]<=row<record['samples'][1]:beta_index=record['beta_index'];break
        result=dict(status='MANO is invalid at this frame')
        torch=self.torch
        if mano_valid:
            beta=np.asarray(data(folder,beta_spec)[beta_index]);pose=np.asarray(data(folder,spec)[row])
            with torch.inference_mode():out=self.layer(torch.tensor(pose.copy())[None],torch.tensor(beta.copy())[None])
            vertices=out.verts[0].cpu().numpy()
            if hand=='left':vertices=vertices*np.array([-1.,1.,1.])
            world=h.get('mano_world_transform') or h['arrays'].get('mano_world_transform')
            placement='stored_world_transform'
            if world:
                transform=np.asarray(data(folder,world)[row])
                if transform.shape==(12,):
                    # Legacy Showee contract: row-major R(9), followed by t(3).
                    r=transform[:9].reshape(3,3);tr=transform[9:]
                else:r=transform[:3,:3];tr=transform[:3,3]
            else:
                from_indices=[191,88,253,708,729,144,87,295,319,220,365,407,445,183,477,518,556,83,589,635,673]
                target=np.asarray(data(folder,h['arrays']['keypoints_world'])[row]);mask=np.asarray(data(folder,h['arrays']['keypoints_valid'])[row],bool)
                if mask.sum()<4:return dict(status='World placement is missing and too few valid points are available')
                r,tr=self.fit(vertices[from_indices][mask],target[mask]);placement='live_Kabsch_preview'
            vertices=vertices@r.T+tr
            result=dict(status='Valid',vertices=vertices,faces=self.faces,pose=pose,beta=beta,placement=placement)
        angle_valid=h['arrays'].get('joint_angles_valid')
        if h['arrays'].get('joint_angles') and (not angle_valid or data(folder,angle_valid)[row]):
            theta=np.asarray(data(folder,h['arrays']['joint_angles'])[row]);torch=self.torch
            if not np.isfinite(theta).all():return result
            with torch.inference_mode():
                angles=torch.tensor(theta.copy())[None];wrist=torch.eye(4)[None]
                fk=self.skin(self.hand,angles,wrist)[0].cpu().numpy()*.001
                model=self.hand
                mesh=self.skin_points(model.joint_rest_positions,model.joint_rotation_axes,
                    model.dense_bone_weights.reshape(1,-1,17),angles,model.mesh_vertices,wrist)[0].cpu().numpy()*.001
            if hand=='right':fk[:,0]*=-1;mesh[:,0]*=-1
            fitted_world=h['arrays'].get('umetrack_world_transform')
            if fitted_world:
                transform=np.asarray(data(folder,fitted_world)[row]);rr=transform[:3,:3];tt=transform[:3,3]
                scale_spec=h['arrays'].get('umetrack_scale')
                scale=float(data(folder,scale_spec)[row]) if scale_spec else 1.
                fk*=scale;mesh*=scale
                result['fk_points']=fk@rr.T+tt;result['fk_placement']='stored CPU-fitted UmeTrack world transform; default hand shape'
                result['umetrack_vertices']=mesh@rr.T+tt;result['umetrack_faces']=self.umetrack_faces
                result['status']='Valid'
            else:
                target=np.asarray(data(folder,h['arrays']['keypoints_world'])[row]);idx=[5,6,7,0,8,9,10,1,11,12,13,2,14,15,16,3,17,18,19,4];mano_idx=[0,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
                good=np.asarray(data(folder,h['arrays']['keypoints_valid'])[row],bool)[mano_idx]&np.isfinite(target[mano_idx]).all(1)
                if good.sum()>=4:
                    rr,tt=self.fit(fk[idx][good],target[mano_idx][good]);result['fk_points']=fk@rr.T+tt;result['fk_placement']='default UmeTrack hand shape, live rigid alignment for preview'
                    result['umetrack_vertices']=mesh@rr.T+tt;result['umetrack_faces']=self.umetrack_faces
                    result['status']='Valid'
        return result

    def render_native_final(self,hand,h,folder,t):
        native=h['native_generated_labels']
        arrays=native['arrays']
        ts,order=time_axis(folder,arrays['timestamps'])
        pos=int(np.searchsorted(ts,t))
        pos=min((max(0,pos-1),min(pos,len(ts)-1)),
                key=lambda i:abs(int(ts[i])-t))
        if abs(int(ts[pos])-t)>50000:
            return dict(status='No nearby native mocap label frame')
        row=int(order[pos]) if order is not None else pos
        key=(str(folder),hand,'native',row)
        if key in self.cache:
            value=self.cache.pop(key);self.cache[key]=value;return value
        result=dict(status='Native mocap is invalid at this frame')
        torch=self.torch
        if data(folder,arrays['mano_valid'])[row]:
            pose=np.asarray(data(folder,arrays['mano_pose'])[row])
            beta=np.asarray(data(folder,arrays['mano_beta'])[row])
            transform=np.asarray(data(folder,arrays['mano_world_transform'])[row])
            with torch.inference_mode():
                out=self.layer(torch.tensor(pose.copy())[None],
                               torch.tensor(beta.copy())[None])
            vertices=out.verts[0].cpu().numpy()
            if hand=='left':vertices[:,0]*=-1
            result.update(status='Valid',
                vertices=vertices@transform[:3,:3].T+transform[:3,3],
                faces=self.faces,pose=pose,beta=beta,
                placement='native_mocap_MANO_fit')
        if data(folder,arrays['joint_angles_valid'])[row]:
            theta=np.asarray(data(folder,arrays['joint_angles'])[row])
            scale=float(data(folder,arrays['umetrack_scale'])[row])
            transform=np.asarray(data(folder,arrays['umetrack_world_transform'])[row])
            with torch.inference_mode():
                angles=torch.tensor(theta.copy())[None]
                wrist=torch.eye(4)[None]
                mesh=self.skin_points(self.hand.joint_rest_positions,
                    self.hand.joint_rotation_axes,
                    self.hand.dense_bone_weights.reshape(1,-1,17),
                    angles,self.hand.mesh_vertices,wrist)[0].cpu().numpy()*.001
            if hand=='right':mesh[:,0]*=-1
            result.update(status='Valid',
                umetrack_vertices=(mesh*scale)@transform[:3,:3].T+transform[:3,3],
                umetrack_faces=self.umetrack_faces,
                fk_placement='native_mocap_UmeTrack_fit')
        self.cache[key]=result
        while len(self.cache)>128:self.cache.popitem(last=False)
        return result

GEOMETRY=None

def render_hand_geometry(hand,h,folder,row,t=None):
    global GEOMETRY
    arrays=h['arrays']
    native=h.get('native_generated_labels')
    if native and (native.get('temporal_interpolation') or h.get('supervision_status')!='native_mocap_labels_published'):
        if GEOMETRY is None:GEOMETRY=Geometry()
        if t is None:t=int(data(folder,arrays['timestamps'])[row])
        return GEOMETRY.render_native_final(hand,h,folder,t)
    if not arrays.get('mano_pose'):return dict(status='MANO/UmeTrack labels are not available')
    valid=arrays.get('mano_valid') or arrays.get('joint_angles_valid') or arrays.get('label_valid')
    angle_valid=arrays.get('joint_angles_valid')
    if valid and not data(folder,valid)[row] and not (
            angle_valid and data(folder,angle_valid)[row]):
        return dict(status='MANO/UmeTrack is invalid at this frame')
    if not arrays.get('mano_beta'):return dict(status='MANO shape parameters are missing')
    if GEOMETRY is None:GEOMETRY=Geometry()
    return GEOMETRY.render(hand,h,folder,row)

def active_annotations(folder,s,t,root):
    labels=[]
    if s.get('interactions'):
        registry=read(root/'objects/index.json') if (root/'objects/index.json').exists() else {'objects':[]}
        names={o['object_id']:o['name'] for o in registry['objects']}
        instances={o['instance_id']:names.get(o['object_id'],o['object_id']) for o in (s.get('objects') or [])}
        for item in read(folder/s['interactions']['path'])['intervals']:
            if item['start_us']<=t<item['end_us']:
                phase={'preparation':'Preparation','interaction':'Interaction'}.get(item['phase'],item['phase'])
                labels.append(phase+' · '+'、'.join(instances.get(x,x) for x in item['object_instances']))
    if not labels:
        vocab=read(root/'dataset.json').get('gesture_classes',[])
        for hand,h in s['hands'].items():
            if not h or not h['arrays'].get('gesture_class'):continue
            row=sample_row(folder,h,t)
            if abs(int(data(folder,h['arrays']['timestamps'])[row])-t)>50000:continue
            index=int(data(folder,h['arrays']['gesture_class'])[row])
            if h['arrays'].get('gesture_active') and not data(folder,h['arrays']['gesture_active'])[row]:continue
            if index<0:continue
            name=vocab.get('classes',{}).get(str(index),{}).get('name') if isinstance(vocab,dict) else (vocab[index] if index<len(vocab) else None)
            if name:labels.append(('Left hand' if hand=='left' else 'Right hand')+' · '+name)
    return labels

@app.get('/api/overlay')
def overlay_data(session:str,device:str,stream:str,time_s:float=0,layers:str='mano',hand:str='both',include_faces:bool=True,frame_index:int|None=None,crop_id:str|None=None):
    """Project selected geometry at the actual decoded camera-frame timestamp."""
    selected=set(layers.split(','))&{'mano','umetrack','markers','phase','bbox'}
    source,e,folder,s=resolve(session)
    if source.get('url'):
        query=dict(session=e['remote_key'],device=device,stream=stream,time_s=time_s,layers=layers,hand=hand,include_faces=str(include_faces).lower())
        if frame_index is not None:query['frame_index']=frame_index
        if crop_id is not None:query['crop_id']=crop_id
        b,_=remote(source['url'],'/api/overlay',query)
        return Response(b,media_type='application/json')
    first,_=span(folder,s);c=s['cameras'][device]['streams'][stream];idx=camera_row(folder,c,int(first+time_s*1e6)) if frame_index is None else frame_index
    if idx is None:raise HTTPException(404,'No valid camera timestamp')
    if not 0<=idx<c['arrays']['timestamps']['shape'][0]:raise HTTPException(404,'Frame outside stream')
    variant=active_crop(c)
    if crop_id is not None and (not variant or variant['crop_id']!=crop_id):raise HTTPException(409,'Crop version changed; refresh')
    t=int(data(folder,c['arrays']['timestamps'])[idx]);root=Path(source['path'])
    record=next(x for x in read(root/'calibrations/index.json') if x['calibration_id']==s['cameras'][device]['calibration_id'])
    intr=load_calibration(root,record)['cameras'][stream]
    result=dict(frame_index=idx,timestamp_us=t,frame_time_s=(t-first)/1e6,image_size=intr['image_size'],hands={},annotations=active_annotations(folder,s,t,root) if 'phase' in selected else [],warnings=[])
    if 'bbox' in selected:
        for side in ('left','right'):
            if hand not in ('both',side):continue
            record=crop_record(folder,c,idx,side)
            result['hands'][side]={'crop':record}
            if not record['valid']:result['warnings'].append(('Left hand' if side=='left' else 'Right hand')+' crop: '+record['reason'])
    if not selected&{'mano','umetrack','markers'}:return fast_json(result)
    if not c['arrays'].get('world_poses') or not c['arrays'].get('pose_valid') or not data(folder,c['arrays']['pose_valid'])[idx]:
        result['warnings'].append('No valid camera-world pose; geometry overlay is unavailable');return fast_json(result)
    transform=np.asarray(data(folder,c['arrays']['world_poses'])[0 if c['world_pose_storage']=='static' else idx]);inv=np.linalg.inv(transform)
    rvec,_=cv2.Rodrigues(inv[:3,:3]);K=np.asarray(intr['K'],float);D=np.asarray(intr['distortion_coefficients'],float)
    def project(points):
        points=np.asarray(points,float)
        if intr['distortion_model']=='equidistant':uv,_=cv2.fisheye.projectPoints(points[None],rvec,inv[:3,3],K,D)
        else:uv,_=cv2.projectPoints(points,rvec,inv[:3,3],K,D)
        z=(points@inv[:3,:3].T+inv[:3,3])[:,2]
        return dict(uv=uv.reshape(-1,2),depth=z)
    for side,h in s['hands'].items():
        if not h or hand not in ('both',side):continue
        row=sample_row(folder,h,t)
        if abs(int(data(folder,h['arrays']['timestamps'])[row])-t)>50000:
            result['warnings'].append(('Left hand' if side=='left' else 'Right hand')+' labels are in a sampling gap');continue
        item=result['hands'].setdefault(side,{})
        if 'markers' in selected:
            native=h.get('native_generated_labels')
            na=native['arrays'] if native and native.get('temporal_interpolation') else None
            if na and na.get('markers_world'):
                nts,order=time_axis(folder,na['timestamps'])
                nr=int(np.searchsorted(nts,t));nr=min((max(0,nr-1),min(nr,len(nts)-1)),key=lambda i:abs(int(nts[i])-t))
                if abs(int(nts[nr])-t)<=50000:
                    nr=int(order[nr]) if order is not None else nr
                    points=np.asarray(data(folder,na['markers_world'])[nr]);item['markers']=dict(project(points),valid=data(folder,na['markers_valid'])[nr])
            elif h['arrays'].get('keypoints_world'):
                points=np.asarray(data(folder,h['arrays']['keypoints_world'])[row]);item['markers']=dict(project(points),valid=data(folder,h['arrays']['keypoints_valid'])[row])
        if selected&{'mano','umetrack'}:
            try:
                with LOCK:
                    geometry=render_hand_geometry(side,h,folder,row,t)
                if 'mano' in selected and geometry.get('vertices') is not None:
                    item['mano']=dict(project(geometry['vertices']),placement=geometry['placement'])
                    if include_faces:item['mano']['faces']=geometry['faces']
                if 'umetrack' in selected and geometry.get('umetrack_vertices') is not None:
                    item['umetrack']=dict(project(geometry['umetrack_vertices']),placement=geometry['fk_placement'])
                    if include_faces:item['umetrack']['faces']=geometry['umetrack_faces']
                if not ('vertices' in geometry or 'umetrack_vertices' in geometry):
                    result['warnings'].append(('Left hand' if side=='left' else 'Right hand')+': '+geometry['status'])
            except Exception as exc:result['warnings'].append(str(exc))
        result['hands'][side]=item
    return fast_json(result)

@app.get('/')
def home():
    html=(HERE/'index.html').read_text(encoding='utf-8')
    for asset in ('app.js','style.css'):
        version=hashlib.sha256((HERE/asset).read_bytes()).hexdigest()[:12]
        html=html.replace('/'+asset,'/'+asset+'?v='+version)
    return HTMLResponse(html,headers={'Cache-Control':'no-store'})

@app.get('/app.js')
def application_script():return FileResponse(HERE/'app.js',media_type='application/javascript',headers={'Cache-Control':'no-store'})

@app.get('/style.css')
def application_styles():return FileResponse(HERE/'style.css',media_type='text/css',headers={'Cache-Control':'no-store'})

def session_date(session_id):
    """Use the capture date encoded in either supported session naming scheme."""
    match = re.search(r'(20\d{2})(\d{2})(\d{2})', session_id)
    return '-'.join(match.groups()) if match else ''


def session_task(session):
    """Classify capture task from explicit metadata or the recorded session type."""
    task = session.get('task_type')
    if task in ('gesture', 'hoi'):
        return task
    name = session['session_id'].lower()
    if 'handobject' in name or re.search(r'(^|_)hoi(_|$)', name):
        return 'hoi'
    if 'midair' in name or 'gesture' in name:
        return 'gesture'
    if session.get('objects') or (session.get('object_coverage') or {}).get('recorded_object_ids'):
        return 'hoi'
    if session.get('source_group') == 'original':
        return 'gesture'
    return 'unknown'


@app.get('/api/catalog')
def catalog():
    result=[]
    for key in SESSIONS:
        source,e,folder,s=resolve(key)
        if source.get('url'):
            row=dict(key=key,session_id=e['session_id'],group=source['id'],remote=True,**e.get('summary',{}))
            row.setdefault('recording_date',session_date(e['session_id']))
            row.setdefault('task_type',session_task(e))
            result.append(row)
        else:
            first,last=span(folder,s);result.append(dict(key=key,session_id=s['session_id'],group=source['id'],remote=False,duration_s=(last-first)/1e6,
                start_us=first,end_us=last,subject=s.get('subject'),hands={k:v['rows'] for k,v in s['hands'].items() if v},
                recording_date=session_date(s['session_id']),task_type=session_task(s)))
    return result

@app.get('/api/calibrations')
def calibration_catalog():
    if COLLECTION is not None:return read(COLLECTION/'calibrations/index.json')
    result=[]
    for source in SOURCES.values():
        if source.get('url'):continue
        root=Path(source['path'])
        for e in read(root/'calibrations/index.json'):
            result.append(dict(dataset_id=source['id'],record=e,calibration=load_calibration(root,e)))
    return clean(result)

@app.get('/api/datasets')
def dataset_metadata():
    result=[]
    for source in SOURCES.values():
        if source.get('url'):
            result.extend(json.loads(remote(source['url'],'/api/datasets',{})[0]))
        else:
            m=read(Path(source['path'])/'dataset.json')
            result.append(dict(dataset_id=source['id'],gesture_classes=m.get('gesture_classes',[]),format_version=m['format_version']))
    return result

@app.get('/api/session')
def session_info(session:str):
    source,e,folder,s=resolve(session)
    if source.get('url'):return json.loads(remote(source['url'],'/api/session',dict(session=e['remote_key']))[0])
    first,last=span(folder,s);videos=[]
    root=Path(source['path'])
    calibration_ids={c['calibration_id'] for c in s['cameras'].values() if c}
    calibrations=[load_calibration(root,e) for e in read(root/'calibrations/index.json') if e['calibration_id'] in calibration_ids]
    objects=read(root/'objects/index.json') if (root/'objects/index.json').exists() else None
    for device,camera in s['cameras'].items():
        for stream,c in camera['streams'].items():
            if c is None:continue
            first_frame_time_s=first_camera_time(folder,c,first)
            videos.append(dict(device=device,stream=stream,first_frame_time_s=first_frame_time_s,available=c['video'].get('available') or bool(c['video'].get('source')),frame_count=c['video']['frame_count'],timing=c.get('timing'),world_pose=c['arrays'].get('world_poses') is not None,hand_crops=active_crop(c)))
    return clean(dict(session=s,calibrations=calibrations,object_registry=objects,gesture_classes=read(root/'dataset.json').get('gesture_classes',[]),start_us=first,end_us=last,duration_s=(last-first)/1e6,videos=videos,
        label_progress=read(Path(source['path'])/'reports/local_label_progress.json') if (Path(source['path'])/'reports/local_label_progress.json').exists() else None))

def trace(folder,h,t,fields=None,window_s=2):
    ts,order=time_axis(folder,h['arrays']['timestamps']);position=row_at(ts,t);row=int(order[position]) if order is not None else position
    half_us=window_s*500000
    begin=max(0,np.searchsorted(ts,t-half_us));end=min(len(ts),np.searchsorted(ts,t+half_us));step=max(1,(end-begin+599)//600);positions=np.arange(begin,end,step);indices=order[positions] if order is not None else positions
    values={};instant={k:np.asarray(data(folder,h['arrays'][k])[row]) for k in ['imu','wrist_angles','joint_angles','mano_residual_mm','mano_valid','joint_angles_valid','label_valid','gesture_class','gesture_active','split_id'] if h['arrays'].get(k)}
    for name in ['emg_raw','emg_filtered','imu','wrist_angles','joint_angles']+[k for k in h['arrays'] if k.endswith('_imu_aligned') or k=='camera_imu_aligned']:
        if fields is not None and name not in fields:continue
        spec=h['arrays'].get(name)
        if spec:
            vals=np.asarray(data(folder,spec)[indices]).copy();valid_name={'emg_raw':'emg_valid','emg_filtered':'emg_valid','imu':'imu_valid','wrist_angles':'wrist_valid','joint_angles':'joint_angles_valid'}.get(name)
            if name=='joint_angles' and not h['arrays'].get(valid_name):valid_name='label_valid'
            if valid_name and h['arrays'].get(valid_name):
                mask=np.asarray(data(folder,h['arrays'][valid_name])[indices],bool)
                if mask.ndim==1:vals[~mask]=np.nan
                else:vals[~mask]=np.nan
            values[name]=vals
            current=np.asarray(data(folder,spec)[row]).copy()
            if valid_name and h['arrays'].get(valid_name):current=np.where(data(folder,h['arrays'][valid_name])[row],current,np.nan)
            if abs(t-int(ts[position]))>50000:current[:]=np.nan
            instant[name]=current
    return dict(row=row,timestamp_us=int(ts[position]),sample_age_us=int(t-ts[position]),time_s=(ts[positions]-t)/1e6,traces=values,instant=instant)

@app.get('/api/state')
def state(session:str,time_s:float=0,geometry:bool=True,fields:str|None=None,native_imu:bool=True,mocap_points:bool=True,window_s:float=Query(2,ge=.2,le=60)):
    source,e,folder,s=resolve(session)
    if source.get('url'):
        query=dict(session=e['remote_key'],time_s=time_s,geometry=str(geometry).lower(),native_imu=str(native_imu).lower(),mocap_points=str(mocap_points).lower(),window_s=window_s)
        if fields is not None:query['fields']=fields
        b,_=remote(source['url'],'/api/state',query)
        return Response(b,media_type='application/json')
    first,last=span(folder,s);t=int(np.clip(first+time_s*1e6,first,last));hands={}
    for hand,h in s['hands'].items():
        if not h:continue
        result=trace(folder,h,t,None if fields is None else set(fields.split(',')),window_s);r=result['row'];kp=h['arrays'].get('keypoints_world')
        native=h.get('native_generated_labels')
        if native and (native.get('temporal_interpolation') or h.get('supervision_status')!='native_mocap_labels_published'):
            arrays=native['arrays']
            ts,order=time_axis(folder,arrays['timestamps']);pos=int(np.searchsorted(ts,t))
            pos=min((max(0,pos-1),min(pos,len(ts)-1)),
                    key=lambda i:abs(int(ts[i])-t))
            native_row=int(order[pos]) if order is not None else pos
            current=abs(int(ts[pos])-t)<=50000
            result['native_generated_validity']=dict(
                mano=bool(current and data(folder,arrays['mano_valid'])[native_row]),
                umetrack=bool(current and data(folder,arrays['joint_angles_valid'])[native_row]))
        if kp:
            result['keypoints']=data(folder,kp)[r];result['keypoint_valid']=data(folder,h['arrays']['keypoints_valid'])[r] if abs(result['sample_age_us'])<=50000 else np.zeros(21,dtype='uint8')
        if geometry:
            with LOCK:
                try:
                    result['geometry']=render_hand_geometry(hand,h,folder,r,t) if abs(result['sample_age_us'])<=50000 else dict(status='Sampling gap: no current hand labels')
                except Exception as exc:result['geometry']=dict(status=str(exc))
        hands[hand]=result
    cameras=[];mocap=None;camera_imu={}
    for device,camera in s['cameras'].items():
        if native_imu and camera.get('imu') and camera['imu'].get('arrays'):
            a=camera['imu']['arrays'];ts=data(folder,a['timestamps']);half_us=window_s*500000;begin=max(0,np.searchsorted(ts,t-half_us));end=min(len(ts),np.searchsorted(ts,t+half_us));step=max(1,(end-begin+599)//600);ids=np.arange(begin,end,step)
            current_row=row_at(ts,t);current=np.asarray(data(folder,a['samples'])[current_row]).copy()
            if abs(t-int(ts[current_row]))>50000:current[:]=np.nan
            if a.get('valid'):current=np.where(data(folder,a['valid'])[current_row],current,np.nan)
            camera_imu[device]=dict(samples=data(folder,a['samples'])[ids],time_s=(ts[ids]-t)/1e6,
                instant=current,timestamp_us=int(ts[current_row]),sample_age_us=int(t-ts[current_row]))
        for stream,c in camera['streams'].items():
            if c is None:continue
            idx=camera_row(folder,c,t);entry=dict(device=device,stream=stream,frame_index=idx,pose_valid=False)
            if idx is not None:entry['sample_age_us']=int(t-data(folder,c['arrays']['timestamps'])[idx])
            if idx is not None and c['arrays'].get('world_poses'):
                ok=bool(data(folder,c['arrays']['pose_valid'])[idx]);entry['pose_valid']=ok
                if ok:entry['world_pose']=data(folder,c['arrays']['world_poses'])[0 if c['world_pose_storage']=='static' else idx]
            cameras.append(entry)
    m=s.get('mocap')
    if mocap_points and m and m.get('arrays',{}).get('points_world'):
        row=row_at(data(folder,m['arrays']['timestamp_us'] if 'timestamp_us' in m['arrays'] else m['arrays']['timestamps']),t)
        mocap=dict(points=data(folder,m['arrays']['points_world'])[row],valid=data(folder,m['arrays']['valid'])[row],names=m['marker_names'])
    intervals=[]
    if s.get('interactions'):
        intervals=[i for i in read(folder/s['interactions']['path'])['intervals'] if i['start_us']<=t<i['end_us']]
    return fast_json(dict(timestamp_us=t,hands=hands,cameras=cameras,camera_imu=camera_imu,mocap=mocap,interactions=intervals,objects=s.get('objects'),missing=s.get('missing',[])))

@app.get('/api/frame')
def frame(session:str,device:str,stream:str,time_s:float=0,overlay:bool=False):
    source,e,folder,s=resolve(session)
    if source.get('url'):
        b,m=remote(source['url'],'/api/frame',dict(session=e['remote_key'],device=device,stream=stream,time_s=time_s,overlay=str(overlay).lower()));return Response(b,media_type=m)
    first,last=span(folder,s);t=int(first+time_s*1e6);c=s['cameras'][device]['streams'][stream];idx=camera_row(folder,c,t)
    if idx is None:raise HTTPException(404,'No valid frame timestamp')
    v=c['video'];decoder=idx
    if v.get('frame_index_map'):decoder=int(data(folder,v['frame_index_map'])[idx])
    if decoder<0:raise HTTPException(404,'Unmapped frame')
    path=folder/v['path'] if v.get('available') else Path(v['source'])
    reader=video_reader(path)
    with reader.lock:
        image=reader.decode(decoder)
        if not overlay:
            if reader.jpeg is None:
                resized=cv2.resize(image,(900,int(image.shape[0]*900/image.shape[1]))) if image.shape[1]>900 else image
                _,encoded=cv2.imencode('.jpg',resized,[cv2.IMWRITE_JPEG_QUALITY,85]);reader.jpeg=encoded.tobytes()
            return Response(reader.jpeg,media_type='image/jpeg',headers={'X-Frame-Index':str(idx)})
        image=image.copy()
    if overlay and c['arrays'].get('world_poses') and c['arrays'].get('pose_valid') and data(folder,c['arrays']['pose_valid'])[idx]:
        calibration=read(Path(source['path'])/'calibrations/index.json');record=next(x for x in calibration if x['calibration_id']==s['cameras'][device]['calibration_id']);intr=load_calibration(Path(source['path']),record)['cameras'][stream]
        transform=np.asarray(data(folder,c['arrays']['world_poses'])[0 if c['world_pose_storage']=='static' else idx]);inv=np.linalg.inv(transform);rvec,_=cv2.Rodrigues(inv[:3,:3]);K=np.asarray(intr['K'],float);D=np.asarray(intr['distortion_coefficients'],float)
        for hand,h in s['hands'].items():
            if not h:continue
            row=sample_row(folder,h,t);points=np.asarray(data(folder,h['arrays']['keypoints_world'])[row]);valid=np.asarray(data(folder,h['arrays']['keypoints_valid'])[row],bool)
            if abs(int(data(folder,h['arrays']['timestamps'])[row])-t)>50000:continue
            if intr['distortion_model']=='equidistant':uv,_=cv2.fisheye.projectPoints(points[None].astype(float),rvec,inv[:3,3],K,D)
            else:uv,_=cv2.projectPoints(points.astype(float),rvec,inv[:3,3],K,D)
            z=(points@inv[:3,:3].T+inv[:3,3])[:,2]
            for xy,good,depth in zip(uv.reshape(-1,2),valid,z):
                if good and depth>0 and np.isfinite(xy).all():cv2.circle(image,tuple(xy.astype(int)),4,(0,200,255) if hand=='left' else (255,160,0),-1)
    if image.shape[1]>900:image=cv2.resize(image,(900,int(image.shape[0]*900/image.shape[1])))
    ok,b=cv2.imencode('.jpg',image,[cv2.IMWRITE_JPEG_QUALITY,85]);return Response(b.tobytes(),media_type='image/jpeg',headers={'X-Frame-Index':str(idx)})

@app.get('/api/hand-crops')
def hand_crop_info(session:str,device:str,stream:str,time_s:float=0,frame_index:int|None=None):
    source,e,folder,s=resolve(session)
    if source.get('url'):
        query=dict(session=e['remote_key'],device=device,stream=stream,time_s=time_s)
        if frame_index is not None:query['frame_index']=frame_index
        b,m=remote(source['url'],'/api/hand-crops',query);return Response(b,media_type=m)
    first,_=span(folder,s);c=s['cameras'][device]['streams'][stream]
    row=camera_row(folder,c,int(first+time_s*1e6)) if frame_index is None else frame_index
    if row is None:raise HTTPException(404,'No camera timestamp')
    if not 0<=row<c['arrays']['timestamps']['shape'][0]:raise HTTPException(404,'Frame outside stream')
    t=int(data(folder,c['arrays']['timestamps'])[row]);variant=active_crop(c)
    progress_path=Path(source['path'])/'reports/hand_crops'/(s['session_id']+'.progress.json')
    generation_progress=read(progress_path) if not variant and progress_path.exists() and (device,stream) in (('gopro','rgb'),('showee_head','camera')) else None
    return fast_json(dict(frame_index=row,timestamp_us=t,frame_time_s=(t-first)/1e6,
        image_size=[256,256],source_image_size=next((x.get('width'),x.get('height')) for x in c['video'].get('clips',[]) ) if c['video'].get('clips') else None,
        crop_id=variant['crop_id'] if variant else None,status=variant.get('status') if variant else 'generating' if generation_progress and generation_progress.get('state')=='running' else 'not_generated',generation_progress=generation_progress,
        hands={hand:crop_record(folder,c,row,hand) for hand in ('left','right')}))

@app.get('/api/hand-crop')
def hand_crop_image(session:str,device:str,stream:str,hand:str,frame_index:int,crop_id:str|None=None):
    source,e,folder,s=resolve(session)
    if source.get('url'):
        query=dict(session=e['remote_key'],device=device,stream=stream,hand=hand,frame_index=frame_index)
        if crop_id is not None:query['crop_id']=crop_id
        b,m,headers=remote(source['url'],'/api/hand-crop',query,with_headers=True);return Response(b,media_type=m,headers=headers)
    if hand not in ('left','right'):raise HTTPException(400,'Unknown hand')
    c=s['cameras'][device]['streams'][stream];variant=active_crop(c)
    if crop_id is not None and (not variant or crop_id!=variant['crop_id']):raise HTTPException(409,'Crop version changed; refresh')
    try:payload,record=crop_bytes(folder,c,frame_index,hand)
    except (IndexError,ValueError) as exc:raise HTTPException(422,str(exc)) from exc
    if payload is None:raise HTTPException(404,record['reason'])
    return Response(payload,media_type='image/jpeg',headers={'X-Frame-Index':str(frame_index),'X-Crop-Hand':hand,'X-Crop-Id':record['crop_id'],'Cache-Control':'private, max-age=60'})

@app.get('/api/depth')
def depth(session:str,device:str,time_s:float=0):
    source,e,folder,s=resolve(session)
    if source.get('url'):
        b,m=remote(source['url'],'/api/depth',dict(session=e['remote_key'],device=device,time_s=time_s));return Response(b,media_type=m)
    d=s['cameras'][device].get('depth')
    if not d:raise HTTPException(404,'Depth not recorded')
    first,last=span(folder,s);t=int(first+time_s*1e6);ts=data(folder,d['timestamps']);mask=np.asarray(data(folder,d['timestamp_valid']),bool)
    if d.get('frame_valid'):mask &= np.asarray(data(folder,d['frame_valid']),bool)
    valid=np.flatnonzero(mask)
    if not len(valid):raise HTTPException(404,'No valid depth frames')
    row=int(valid[row_at(ts[valid],t)])
    if abs(int(ts[row])-t)>100000:raise HTTPException(404,'Depth stream unavailable in this interval')
    if not continuity_valid(folder,d,row,t):raise HTTPException(404,'Depth unavailable across cut seam')
    native_row=int(data(folder,d['native_frame_index'])[row]) if d.get('native_frame_index') else row
    sys.path.insert(0,str(REPO));from scripts.tools.depth_rvl import open_depth
    with LOCK,open_depth(folder/d['path']) as f:values=f[d['dataset']][native_row]
    meters=values.astype('float32')*d['scale_to_meters'];norm=(np.clip(meters,0,3)/3*255).astype('uint8');image=cv2.applyColorMap(255-norm,cv2.COLORMAP_TURBO);image[values==0]=0
    image=cv2.resize(image,(640,int(image.shape[0]*640/image.shape[1])));ok,b=cv2.imencode('.jpg',image)
    return Response(b.tobytes(),media_type='image/jpeg',headers={'X-Depth-Frame':str(row),'X-Depth-Unit':'metres'})

def register(source):
    SOURCES[source['id']]=source
    if source.get('url'):
        try:rows=json.loads(remote(source['url'],'/api/catalog',{},timeout=3)[0])
        except Exception:rows=source.get('sessions',[])
        for e in rows:
            if source.get('session_ids') is not None and e['session_id'] not in source['session_ids']:continue
            sid=e['session_id'];SESSIONS[source['id']+'/'+sid]=(source,dict(session_id=sid,remote_key=e['key'],summary={k:e[k] for k in ['duration_s','start_us','end_us','subject','hands','recording_date','task_type'] if k in e}))
    else:
        for e in read(Path(source['path'])/'dataset.json')['sessions']:
            if source.get('session_ids') is not None and e['session_id'] not in source['session_ids']:continue
            SESSIONS[source['id']+'/'+e['session_id']]=(source,e)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--collection');p.add_argument('--dataset',action='append',default=[]);p.add_argument('--port',type=int,default=8787);a=p.parse_args()
    if a.collection:
        COLLECTION=Path(a.collection).resolve().parent
        for source in read(a.collection)['datasets']:
            if source.get('path') and not Path(source['path']).is_absolute():source['path']=str(Path(a.collection).resolve().parent/source['path'])
            register(source)
    for spec in a.dataset:
        name,path=spec.split('=',1);register(dict(id=name,path=path))
    uvicorn.run(app,host='127.0.0.1',port=a.port)
