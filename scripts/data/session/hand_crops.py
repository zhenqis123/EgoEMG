"""Session-native JPEG byte storage, crop geometry, exporter and shared reader (no LMDB)."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
import re
from pathlib import Path
import time

import cv2
import numpy as np
# Support direct script execution from a source checkout.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from egoemg.data_processing.calibration_io import read_json, load_calibration

PROFILE = dict(name='ego_hand_square_256_v2', patch_size=256, jpeg_quality=95,
               chroma_subsampling='444', horizontal_flip=[False, False], canonical_hand=None,
               bbox_margin=1.44, previous_profile='ego_hand_square_256_v1', side_expansion_from_previous=1.2,
               minimum_points=3, minimum_side_px=12,
               minimum_image_overlap=.25, maximum_sample_age_us=50000,
               interpolation='opencv_linear', border='constant_black', locator='world_markers')
REASONS = {0:'valid',1:'source_frame_or_time_unavailable',2:'camera_pose_unavailable',
           3:'insufficient_points',4:'crop_range_invalid',5:'image_generation_failed',
           6:'legacy_source_unverified',7:'not_processed'}

def read(path):return read_json(path)
def write(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.'+str(os.getpid())+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');temp.replace(path)
def array(folder,spec):return np.memmap(Path(folder)/spec['path'],dtype=spec['dtype'],shape=tuple(spec['shape']),mode='r')
def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()
def fingerprint(path):
    path=Path(path);st=path.stat()
    return dict(bytes=st.st_size,mtime_ns=st.st_mtime_ns)
def active(stream):
    group=stream.get('hand_crops') or {}
    return next((x for x in group.get('variants',[]) if x['crop_id']==group.get('active_id')),None)

def crop_record(folder,stream,row,hand):
    if hand not in ('left','right'):raise ValueError('Invalid hand')
    variant=active(stream)
    if not variant:return dict(valid=False,reason='crop_not_generated',hand=hand,frame_index=row)
    i=0 if hand=='left' else 1;a=variant['arrays']
    if row<0 or row>=a['valid']['shape'][0]:raise IndexError(row)
    valid=bool(array(folder,a['valid'])[row,i]);reason=int(array(folder,a['invalid_reason'])[row,i])
    result=dict(valid=valid,reason=REASONS.get(reason,'unknown'),hand=hand,frame_index=row,
                bbox=np.asarray(array(folder,a['bbox_xyxy'])[row,i]).tolist() if valid else None,
                source_to_crop=np.asarray(array(folder,a['source_to_crop'])[row,i]).tolist() if valid else None,
                crop_id=variant['crop_id'],geometry_quality=variant.get('geometry_quality','verified_dataset_geometry'))
    if valid:
        offset,length=(int(x) for x in array(folder,a['offsets'])[row,i])
        path=Path(folder)/variant['storage']['path']
        if length<=0 or offset+length>path.stat().st_size:raise ValueError('Crop byte range invalid')
        result.update(byte_offset=offset,byte_length=length)
    return result

def crop_bytes(folder,stream,row,hand):
    record=crop_record(folder,stream,row,hand)
    if not record['valid']:return None,record
    with (Path(folder)/active(stream)['storage']['path']).open('rb') as f:
        f.seek(record['byte_offset']);payload=f.read(record['byte_length'])
    if len(payload)!=record['byte_length']:raise ValueError('Truncated crop')
    return payload,record

def read_crop(folder,stream,row,hand):
    payload,record=crop_bytes(folder,stream,row,hand)
    if payload is None:return None,record
    image=cv2.imdecode(np.frombuffer(payload,np.uint8),cv2.IMREAD_COLOR)
    if image is None or image.shape!=(256,256,3):raise ValueError('Invalid encoded crop image')
    return cv2.cvtColor(image,cv2.COLOR_BGR2RGB),record

@np.errstate(invalid="ignore",divide="ignore")
def geometry(camera_points,point_valid,intr):
    """Vectorized source-pixel bboxes/affines; never flip either hand."""
    points=np.asarray(camera_points,dtype=np.float64);mask=np.asarray(point_valid,bool)&np.isfinite(points).all(-1)&(points[...,2]>.01)
    safe=points.copy();safe[~mask]=[0,0,1]
    K=np.asarray(intr['K'],float);D=np.asarray(intr['distortion_coefficients'],float)
    if intr['distortion_model']=='equidistant':pixels=cv2.fisheye.projectPoints(safe.reshape(-1,1,3),np.zeros(3),np.zeros(3),K,D)[0]
    elif intr['distortion_model'] in ('opencv','none','brown_conrady','opencv_rectified'):
        pixels=cv2.projectPoints(safe.reshape(-1,3),np.zeros(3),np.zeros(3),K,D if intr['distortion_model'] not in ('none','opencv_rectified') else None)[0]
    else:raise ValueError('Unsupported crop projection model: '+intr['distortion_model'])
    pixels=pixels.reshape(*points.shape[:-1],2);mask &= np.isfinite(pixels).all(-1)
    lo=np.min(np.where(mask[...,None],pixels,np.inf),axis=1);hi=np.max(np.where(mask[...,None],pixels,-np.inf),axis=1)
    enough=mask.sum(1)>=PROFILE['minimum_points']
    side=np.max(hi-lo,axis=1)*PROFILE['bbox_margin'];center=(lo+hi)*.5
    bbox=np.concatenate([center-side[:,None]*.5,center+side[:,None]*.5],axis=1)
    width,height=intr['image_size']
    overlap=np.maximum(0,np.minimum(bbox[:,2],width)-np.maximum(bbox[:,0],0))*np.maximum(0,np.minimum(bbox[:,3],height)-np.maximum(bbox[:,1],0))
    good=enough&np.isfinite(bbox).all(1)&(side>=PROFILE['minimum_side_px'])&(overlap/np.maximum(side*side,1)>=PROFILE['minimum_image_overlap'])
    affine=np.zeros((len(points),3,3),np.float32);affine[good,0,0]=PROFILE['patch_size']/side[good];affine[good,1,1]=affine[good,0,0]
    affine[good,0,2]=-bbox[good,0]*affine[good,0,0];affine[good,1,2]=-bbox[good,1]*affine[good,1,1];affine[good,2,2]=1
    reason=np.where(enough,4,3).astype(np.uint8);reason[good]=0;bbox[~good]=0
    return bbox.astype(np.float32),affine,good,reason

def jpeg_444(payload):
    """Validate JPEG SOF sampling, independent of the encoder flag."""
    i=2
    while i<len(payload)-4:
        if payload[i]!=255:raise ValueError('Malformed JPEG')
        marker=payload[i+1];i+=2
        if marker in (0xD8,0xD9):continue
        n=int.from_bytes(payload[i:i+2],'big')
        if marker in (0xC0,0xC1,0xC2):
            components=payload[i+7]
            return components==3 and all(payload[i+9+j*3]==0x11 for j in range(components))
        i+=n
    return False

def native_marker_groups(names,hand):
    joints={'HandM','HandThumb','ThumbM1','ThumbM2','ThumbM3','HandIndex','IndexM1','IndexM2','IndexM3','HandMiddle','MiddleM1','MiddleM2','MiddleM3','HandRing','RingM1','RingM2','RingM3','HandPinky','PinkyM1','PinkyM2','PinkyM3'}
    systems={}
    for i,name in enumerate(names):
        match=re.fullmatch(('Left' if hand=='left' else 'Right')+r'HandSys(\d+)(.+)',name)
        if not match:continue
        joint='HandPinky' if match[2]=='HandLittle' else match[2]
        if joint not in joints:continue
        group=systems.setdefault(match[1],{})
        if joint not in group or match[2]==joint:group[joint]=i
    columns=[];groups=[]
    for group_index,group in enumerate(systems.values()):
        for column in group.values():columns.append(column);groups.append(group_index)
    return np.asarray(columns,np.int64),np.asarray(groups,np.int64)

def legacy_context(session,root):
    if not root or session.get('source',{}).get('format')!='egoemg_v3_memmap':return None
    root=Path(root);manifest=read(root/'manifest.json');metadata=np.load(root/'metadata.npz',allow_pickle=False)
    ep=session['source']['episode_index'];begin,end=session['source']['source_row_interval']
    if int(metadata['episode_start_idx'][ep])!=begin or int(metadata['episode_end_idx'][ep])!=end:raise ValueError('Legacy episode row provenance mismatch')
    fields={name:np.memmap(root/spec['filename'],dtype=spec['dtype'],shape=tuple(spec['shape']),mode='r') for name,spec in manifest['fields'].items()
            if name in ('image_head_frame_index','image_head_delta_ms','image_head_stale','timestamp_us','mocap_head_transform','mocap_head_valid')}
    frames=np.asarray(fields['image_head_frame_index'][begin:end]);valid_rows=np.flatnonzero(~np.asarray(fields['image_head_stale'][begin:end],bool));unique,first=np.unique(frames[valid_rows],return_index=True);first=valid_rows[first]
    keep=unique>=0;lookup=np.full(int(unique[keep].max())+1,-1,np.int64);lookup[unique[keep]]=first[keep]+begin
    return dict(fields=fields,lookup=lookup,begin=begin,root=root,
                provenance='Historical optical camera transforms used only for crop derivation; world-extrinsics provenance remains unresolved')

def reconstruct_camera_clock(sample_times,frames,delta_ms,stale,count):
    lo=np.full(count,np.iinfo(np.int64).max,np.int64);hi=np.zeros(count,np.int64)
    for begin in range(0,len(frames),65536):
        idx=np.asarray(frames[begin:begin+65536],np.int64);mask=(idx>=0)&(idx<count)&~np.asarray(stale[begin:begin+65536],bool)
        estimated=np.asarray(sample_times[begin:begin+65536],np.int64)[mask]+np.asarray(delta_ms[begin:begin+65536],np.int64)[mask]*1000
        np.minimum.at(lo,idx[mask],estimated);np.maximum.at(hi,idx[mask],estimated)
    valid=(hi>0)&(lo<=hi)&((hi-lo)<=20000);times=np.zeros(count,np.int64);times[valid]=lo[valid]+(hi[valid]-lo[valid])//2
    return times,valid

def repair_legacy_head_clock(folder,session,c,legacy):
    if c.get('timestamp_delta_definition')=='frame_time_minus_sample_time':return
    n=c['arrays']['timestamps']['shape'][0];begin,end=session['source']['source_row_interval'];a=legacy['fields']
    times,valid=reconstruct_camera_clock(a['timestamp_us'][begin:end],a['image_head_frame_index'][begin:end],a['image_head_delta_ms'][begin:end],a['image_head_stale'][begin:end],n)
    old_valid=int(array(folder,c['arrays']['timestamp_valid']).sum())
    if int(valid.sum())<old_valid:raise ValueError('Clock sign correction reduced coverage')
    c['legacy_timestamp_backup']={'arrays':{k:c['arrays'][k] for k in ('timestamps','timestamp_valid')},'timing':c.get('timing'),'reason':'previous exporter subtracted frame-minus-sample delta'}
    for name,value in [('timestamps',times),('timestamp_valid',valid.astype('uint8'))]:
        path=Path('cameras/showee_head/camera')/(name+'_delta_sign_corrected.dat');value.tofile(folder/path)
        c['arrays'][name]=dict(path=path.as_posix(),dtype=value.dtype.str,shape=list(value.shape),time_axis='showee_head/camera')
    c['timestamp_delta_definition']='frame_time_minus_sample_time'
    c['timing']='Legacy frame UTC reconstructed as sample_timestamp_us + delta_ms * 1000; integer-ms quantization; not calibrated exposure'
    session_path=folder/'session.json';latest=read(session_path);latest['cameras']['showee_head']['streams']['camera']=c;write(session_path,latest)
    return dict(old_valid=old_valid,new_valid=int(valid.sum()),frames=n)

def prepare(root,entry,limit=None,legacy_root=None,decode_threads=8):
    start=time.time();root=Path(root);session_path=root/entry['path'];folder=session_path.parent;s=read(session_path)
    if not any(name in s['cameras'] for name in ('gopro','showee_head')):
        return dict(session_id=s['session_id'],skipped=True,reason='ego_rgb_not_recorded',images=0,bytes=0)
    device='gopro' if 'gopro' in s['cameras'] else 'showee_head';stream_name='rgb' if device=='gopro' else 'camera'
    camera=s['cameras'][device];c=camera['streams'].get(stream_name)
    if not c or not c.get('arrays',{}).get('timestamps') or not c['arrays']['timestamps']['shape'][0]:
        return dict(session_id=s['session_id'],skipped=True,reason='ego_rgb_not_recorded',images=0,bytes=0)
    n=c['arrays']['timestamps']['shape'][0]
    count=min(n,limit) if limit else n
    index=read(root/'calibrations/index.json');record=next(x for x in index if x['calibration_id']==camera['calibration_id'])
    intr=load_calibration(root,record)['cameras'][stream_name]
    v=c['video'];video=folder/v['path'] if v.get('available') else Path(v.get('source',''))
    if not video.is_file():raise FileNotFoundError(str(video))
    native=s.get('mocap') if any(h is None for h in s['hands'].values()) else None
    legacy=legacy_context(s,legacy_root) if not c['arrays'].get('world_poses') else None
    clock_repair=repair_legacy_head_clock(folder,s,c,legacy) if legacy else None
    dependencies={'profile':PROFILE,'intrinsics':intr,'source_video':fingerprint(video),
                  'camera_arrays':{k:fingerprint(folder/a['path']) for k,a in c['arrays'].items() if a},
                  'hand_arrays':{hand:{k:fingerprint(folder/a['path']) for k,a in h['arrays'].items() if k in ('timestamps','keypoints_world','keypoints_valid','continuity_segment')} for hand,h in s['hands'].items() if h},
                  'frame_index_map':fingerprint(folder/v['frame_index_map']['path']) if v.get('frame_index_map') else None,'limit':limit}
    if native:
        dependencies['native_marker_fallback']={'arrays':{k:fingerprint(folder/a['path']) for k,a in native['arrays'].items()},'marker_names':native['marker_names']}
    if legacy:dependencies['legacy_geometry']={k:fingerprint(legacy['root']/read(legacy['root']/'manifest.json')['fields'][k]['filename']) for k in legacy['fields']}
    crop_id='ego_hand_'+hashlib.sha256(json.dumps(dependencies,sort_keys=True).encode()).hexdigest()[:16]
    existing=active(c)
    if existing and existing['crop_id']==crop_id and existing.get('generation_complete'):
        return dict(session_id=s['session_id'],skipped=True,**existing['summary'])
    base=folder/'cameras'/device/stream_name/'hand_crops';stage=base/(crop_id+'.building');final=base/crop_id
    if stage.exists():
        # Only remove this version's incomplete, explicitly resolved staging directory.
        import shutil
        if stage.resolve().parent!=base.resolve() or not stage.name.endswith('.building'):raise ValueError('Unsafe staging path')
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    shapes={'offsets':('<u8',(n,2,2)),'valid':('u1',(n,2)),'invalid_reason':('u1',(n,2)),
            'bbox_xyxy':('<f4',(n,2,4)),'source_to_crop':('<f4',(n,2,3,3))}
    maps={key:np.memmap(stage/(key+'.dat'),dtype=dtype,shape=shape,mode='w+') for key,(dtype,shape) in shapes.items()}
    for key,m in maps.items():m[:]=7 if key=='invalid_reason' else 0
    times=array(folder,c['arrays']['timestamps']);timestamp_valid=array(folder,c['arrays']['timestamp_valid']) if c['arrays'].get('timestamp_valid') else times>0
    camera_segments=array(folder,c['arrays']['continuity_segment']) if c['arrays'].get('continuity_segment') else None
    if c['arrays'].get('frame_valid'):
        timestamp_valid=np.asarray(timestamp_valid,bool)&np.asarray(array(folder,c['arrays']['frame_valid']),bool)
    poses=array(folder,c['arrays']['world_poses']) if c['arrays'].get('world_poses') else None
    pose_valid=array(folder,c['arrays']['pose_valid']) if c['arrays'].get('pose_valid') else None
    frame_map=array(folder,v['frame_index_map']) if v.get('frame_index_map') else np.arange(n)
    target_rows={int(dec):row for row,dec in enumerate(frame_map[:count]) if dec>=0}
    if len(target_rows)!=np.count_nonzero(np.asarray(frame_map[:count])>=0):raise ValueError('Nonunique physical frame mapping')
    hand_data={}
    for hand,h in s['hands'].items():
        if not h:continue
        a=h['arrays'];ts=array(folder,a['timestamps']);order=None
        if np.any(np.diff(ts)<0):order=np.argsort(ts,kind='stable');ts=ts[order]
        rows=np.clip(np.searchsorted(ts,times[:count],side='right')-1,0,len(ts)-1);ages=np.abs(np.asarray(ts[rows],np.int64)-times[:count]);rows=order[rows] if order is not None else rows
        hand_data[hand]=dict(points=array(folder,a['keypoints_world']),valid=array(folder,a['keypoints_valid']),rows=rows,ages=ages,
            segments=array(folder,a['continuity_segment']) if a.get('continuity_segment') else None)
    if native:
        if native.get('unit')!='m':raise ValueError('Native marker fallback requires metre coordinates')
        na=native['arrays'];ts=array(folder,na.get('timestamps',na.get('timestamp_us')));order=np.argsort(ts,kind='stable') if np.any(np.diff(ts)<0) else None
        ordered_ts=ts[order] if order is not None else ts
        positions=np.clip(np.searchsorted(ordered_ts,times[:count],side='right')-1,0,len(ts)-1)
        for hand,h in s['hands'].items():
            if h is not None:continue
            columns,groups=native_marker_groups(native['marker_names'],hand)
            if not len(columns):continue
            hand_data[hand]=dict(points=array(folder,na['points_world']),valid=array(folder,na['valid']),
                rows=order[positions] if order is not None else positions,ages=np.abs(ordered_ts[positions]-times[:count]),columns=columns,groups=groups,source='native_mocap_without_emg')
    digest=hashlib.sha256();projected=0
    for begin in range(0,count,2048):
        end=min(begin+2048,count);ids=np.arange(begin,end);good_time=np.asarray(timestamp_valid[ids],bool)
        if poses is not None:
            transform=np.asarray(poses[np.zeros(len(ids),np.int64) if c.get('world_pose_storage')=='static' else ids],float)
            pose_ok=np.asarray(pose_valid[ids],bool) if pose_valid is not None else np.zeros(len(ids),bool)
            legacy_rows=None
        elif legacy:
            lookup=legacy['lookup'];legacy_rows=np.full(len(ids),-1,np.int64);allowed=ids<len(lookup);legacy_rows[allowed]=lookup[ids[allowed]]
            pose_ok=legacy_rows>=0;raw=np.asarray(legacy['fields']['mocap_head_transform'][np.maximum(legacy_rows,0)],float)
            transform=np.tile(np.eye(4),(len(ids),1,1));transform[:,:3,:3]=raw[:,:9].reshape(-1,3,3);transform[:,:3,3]=raw[:,9:12]
            if 'mocap_head_valid' in legacy['fields']:pose_ok &= np.asarray(legacy['fields']['mocap_head_valid'][np.maximum(legacy_rows,0)],bool)
        else:
            transform=np.tile(np.eye(4),(len(ids),1,1));pose_ok=np.zeros(len(ids),bool);legacy_rows=None
        rotation=transform[:,:3,:3];pose_ok &= np.isfinite(transform).all((1,2))&(np.abs(np.linalg.det(rotation)-1)<1e-3)
        for hi,hand in enumerate(('left','right')):
            hd=hand_data.get(hand)
            if hd is None:maps['invalid_reason'][ids,hi]=3;continue
            rows=hd['rows'][ids]
            if legacy_rows is not None:rows=np.clip(legacy_rows-legacy['begin'],0,len(hd['points'])-1)
            world=np.asarray(hd['points'][rows],float);mask=np.asarray(hd['valid'][rows],bool)
            if 'columns' in hd:
                world=world[:,hd['columns']];mask=mask[:,hd['columns']]
                groups=hd['groups'];tracked=np.column_stack([mask[:,groups==group].sum(1)>=PROFILE['minimum_points'] for group in np.unique(groups)])
                mask[tracked.sum(1)!=1]=False
                for group in np.unique(groups):mask[:,groups==group]&=tracked[:,group,None]

            digest.update(np.ascontiguousarray(world).tobytes());digest.update(np.ascontiguousarray(mask).tobytes());digest.update(np.ascontiguousarray(transform).tobytes());digest.update(np.ascontiguousarray(times[ids]).tobytes())
            cam=np.einsum('fpi,fij->fpj',world-transform[:,None,:3,3],rotation)
            bbox,affine,good,reason=geometry(cam,mask,intr)
            reason[~pose_ok]=2;age_ok=hd['ages'][ids]<=PROFILE['maximum_sample_age_us']
            if camera_segments is not None and hd.get('segments') is not None:
                age_ok &= hd['segments'][rows] == camera_segments[ids]
            if legacy_rows is not None:age_ok=legacy_rows>=0
            reason[~good_time|~age_ok]=1;good &= good_time&pose_ok&age_ok
            maps['bbox_xyxy'][ids,hi]=np.where(good[:,None],bbox,0);maps['source_to_crop'][ids,hi]=np.where(good[:,None,None],affine,0)
            maps['invalid_reason'][ids,hi]=reason;maps['valid'][ids,hi]=good
        projected=end
    if not hasattr(cv2,'IMWRITE_JPEG_SAMPLING_FACTOR'):raise RuntimeError('Encoder cannot explicitly select JPEG 444')
    import av
    params=[cv2.IMWRITE_JPEG_QUALITY,95,cv2.IMWRITE_JPEG_SAMPLING_FACTOR,cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444]
    written=0;decoded=0;verified=0;data_hash=hashlib.sha256();data_path=stage/'images.data';job=root/'reports/hand_crops';progress_path=job/(s['session_id']+'.progress.json')
    # CPU sequential decoding measured faster than CUDA host transfers for this crop workload.
    decode_options={}
    try:container=av.open(str(video),**decode_options)
    except (ValueError,RuntimeError):container=av.open(str(video))
    with data_path.open('wb') as out,container:
        stream=container.streams.video[0]
        stream.codec_context.thread_count=max(1,int(decode_threads))
        stream.codec_context.thread_type='FRAME'
        for ordinal,frame in enumerate(container.decode(stream)):
            decoded=ordinal+1
            if limit and ordinal>max(target_rows,default=-1):break
            row=target_rows.get(ordinal)
            if row is None:continue
            if frame.width!=intr['image_size'][0] or frame.height!=intr['image_size'][1]:raise ValueError('Source pixel domain size mismatch')
            if not maps['valid'][row].any():continue
            image=frame.to_ndarray(format='bgr24')
            for hi in range(2):
                if not maps['valid'][row,hi]:continue
                patch=cv2.warpAffine(image,maps['source_to_crop'][row,hi,:2],(256,256),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
                ok,jpeg=cv2.imencode('.jpg',patch,params)
                if not ok:raise RuntimeError('JPEG encoding failed')
                payload=jpeg.tobytes()
                if written<4 or written%10000==0:
                    if not jpeg_444(payload):raise ValueError('JPEG chroma sampling mismatch')
                    decoded_patch=cv2.imdecode(jpeg,cv2.IMREAD_COLOR)
                    if decoded_patch is None or decoded_patch.shape!=(256,256,3):raise ValueError('JPEG verification failed')
                    verified+=1
                maps['offsets'][row,hi]=[out.tell(),len(payload)];out.write(payload);data_hash.update(payload);written+=1
            if row%5000==0:write(progress_path,dict(state='running',session=s['session_id'],frames_done=row+1,frames_total=count,images=written,seconds=time.time()-start,pid=os.getpid()))
        out.flush();os.fsync(out.fileno())
    missed=(maps['valid'][:count]!=0)&(maps['offsets'][:count,:,1]==0);maps['valid'][:count][missed]=0;maps['invalid_reason'][:count][missed]=1;maps['bbox_xyxy'][:count][missed]=0;maps['source_to_crop'][:count][missed]=0
    if not limit and decoded!=int(v['frame_count']):raise ValueError(f'Decoded frame count mismatch {decoded} != {v["frame_count"]}')
    summary=dict(frames=n,processed_frames=count,images=written,coverage_per_hand=[float(maps['valid'][:,h].sum())/n for h in range(2)],hardware_decode=bool(getattr(stream.codec_context,'is_hwaccel',False)),decode_threads=max(1,int(decode_threads)),decode_thread_type='FRAME',valid_per_hand=[int(maps['valid'][:,h].sum()) for h in range(2)],bytes=data_path.stat().st_size,seconds=round(time.time()-start,2),jpeg_probes=verified,
                 reasons={str(i):int((maps['invalid_reason']==i).sum()) for i in REASONS})
    specs={}
    for key,m in maps.items():
        m.flush();specs[key]=dict(path=(final/(key+'.dat')).relative_to(folder).as_posix(),dtype=np.dtype(shapes[key][0]).str,shape=list(shapes[key][1]),time_axis=device+'/'+stream_name,sha256=sha(stage/(key+'.dat')))
    specs['offsets']['unit']='byte';specs['bbox_xyxy']['unit']='px'
    specs['source_to_crop']['semantics']='native decoded video pixels to crop pixels; actual affine; no mirroring'
    maps.clear()
    storage=dict(format='encoded_image_bytes',codec='jpeg',path=(final/'images.data').relative_to(folder).as_posix(),jpeg_quality=95,chroma_subsampling='444',image_size=[256,256],bytes=summary['bytes'],sha256=data_hash.hexdigest())
    variant=dict(crop_id=crop_id,status='partial' if limit else 'ready',generation_complete=not bool(limit),profile=PROFILE,source_time_axis=device+'/'+stream_name,
                 hand_order=['left','right'],horizontal_flip=[False,False],canonical_hand=None,arrays=specs,storage=storage,summary=summary,
                 geometry_quality='historical_projection_provenance_unresolved' if legacy else 'dataset_camera_geometry',
                 provenance=dict(dependencies=dependencies,geometry_inputs_sha256=digest.hexdigest(),source_video_sha256=sha(video),
                                 legacy_geometry_note=legacy['provenance'] if legacy else None,clock_repair=clock_repair,marker_sources={hand:hd.get('source','emg_aligned_markers') for hand,hd in hand_data.items()}))
    write(stage/'manifest.json',variant)
    del m
    if final.exists():raise FileExistsError('Immutable crop version already exists: '+str(final))
    stage.replace(final)
    latest=read(session_path);sc=latest['cameras'][device]['streams'][stream_name];group=sc.setdefault('hand_crops',dict(schema_version=1,active_id=None,variants=[]))
    group['variants'].append(variant);group['active_id']=crop_id;write(session_path,latest)
    result=dict(session_id=s['session_id'],crop_id=crop_id,geometry_quality=variant['geometry_quality'],**summary)
    write(job/(s['session_id']+'.json'),result);write(progress_path,dict(state='complete',**result));return result

def run(root,workers=3,session_ids=None,limit=None,legacy_root=None):
    root=Path(root);entries=[e for e in read(root/'dataset.json')['sessions'] if not session_ids or e['session_id'] in session_ids]
    job=root/'reports/hand_crops';results=[];errors=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        threads=max(1,min(8,max(1,(os.cpu_count() or 4)-4)//max(1,workers)))
        futures={pool.submit(prepare,str(root),e,limit,legacy_root,threads):e for e in entries}
        for f in as_completed(futures):
            try:results.append(f.result());print(json.dumps(results[-1]),flush=True)
            except Exception as exc:errors.append(dict(session=futures[f]['session_id'],error=repr(exc)));print(json.dumps(errors[-1]),flush=True)
            write(job/'progress.json',dict(state='running',completed=len(results),total=len(entries),errors=errors,images=sum(r.get('images',0) for r in results)))
    report=dict(state='failed' if errors else 'complete',sessions=len(entries),results=results,errors=errors,images=sum(r.get('images',0) for r in results),bytes=sum(r.get('bytes',0) for r in results))
    write(job/('selection_report.json' if session_ids else 'report.json'),report);write(job/'progress.json',report)
    if errors:raise RuntimeError('Crop jobs failed: '+str(errors))
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--workers',type=int,default=3);p.add_argument('--session',action='append');p.add_argument('--limit',type=int);p.add_argument('--legacy-root')
    a=p.parse_args();run(a.dataset,a.workers,a.session,a.limit,a.legacy_root)
