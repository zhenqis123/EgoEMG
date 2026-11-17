"use strict";
const $=id=>document.getElementById(id);
const TASKS={gesture:'Gesture',hoi:'HOI',unknown:'Unclassified'};
const COLORS={left:'#35bea9',right:'#f3b16d'};
const SIGNALS={emg:{name:'EMG',description:'8-channel raw or session zero-phase FFT EMG',field:'emg_filtered',unit:'mV'},wrist:{name:'Wrist angles',description:'Pitch / yaw · both wrists',field:'wrist_angles',unit:'deg (°)'},imu:{name:'Wristband IMU',description:'Acceleration and angular velocity',field:'imu',unit:'m/s² · rad/s'},joint:{name:'Finger angles',description:'20 UmeTrack finger joint angles',field:'joint_angles',unit:'rad'},cameraIMU:{name:'Camera IMU',description:'Native camera IMU and hand-aligned samples',field:'camera_imu_aligned',unit:'m/s² · rad/s'}};
const OVERLAYS={mano:'MANO mesh',umetrack:'UmeTrack mesh',markers:'Hand markers',bbox:'Hand crop boxes',phase:'Phase / action'};
const UI={catalog:[],selected:'',metadata:null,panels:[],time:0,playing:false,token:0,nextId:1,state:null,stateBusy:false,lastState:0,lastTick:performance.now(),loading:false};
const visibility=new IntersectionObserver(entries=>{let entered=false;for(let e of entries){let p=UI.panels.find(p=>p.node===e.target);if(!p)continue;let before=p.visible;p.visible=e.isIntersecting;if(p.visible&&!before){p.lastRequest=0;entered=true}}if(entered)refresh(true)},{root:$('workspace'),rootMargin:'150px 0px'});
function element(tag,cls,text){let e=document.createElement(tag);if(cls)e.className=cls;if(text!==undefined)e.textContent=text;return e}
function formatTime(s){s=Math.max(0,s||0);return String(Math.floor(s/60)).padStart(2,'0')+':'+String(Math.floor(s%60)).padStart(2,'0')}
function dateLabel(row){let m=row.session_id.match(/(20\d{6})(?:_(\d{6}))?/);if(!m)return row.session_id;let d=m[1],text=d.slice(4,6)+'-'+d.slice(6,8)+'';if(m[2])text+=' '+m[2].slice(0,2)+':'+m[2].slice(2,4);else if(row.start_us)text+=' '+new Date(row.start_us/1000).toLocaleTimeString('en-GB',{hour:'2-digit',minute:'2-digit',hour12:false});return text}
function recordingDate(row){
    if(row.recording_date)return row.recording_date;
    const match=row.session_id.match(/(20\d{2})(\d{2})(\d{2})/);
    return match?match[1]+'-'+match[2]+'-'+match[3]:'Unknown date';
}
function taskType(row){
    if(TASKS[row.task_type])return row.task_type;
    if(/handobject|(?:^|_)hoi(?:_|$)/i.test(row.session_id))return 'hoi';
    if(/midair|gesture/i.test(row.session_id))return 'gesture';
    return 'unknown';
}
function compareSessions(a,b){
    const date=recordingDate(a).localeCompare(recordingDate(b));
    const rank={gesture:0,hoi:1,unknown:2};
    const start=row=>Number.isFinite(row.start_us)?row.start_us:Infinity;
    return date||rank[taskType(a)]-rank[taskType(b)]||start(a)-start(b)||a.session_id.localeCompare(b.session_id)||a.key.localeCompare(b.key);
}
async function api(path,params={},signal){let r=await fetch(path+'?'+new URLSearchParams(params),{signal});if(!r.ok){let detail=await r.text();throw Error(r.status===500?'Server read failed; check the dataset or remote connection':detail)}return r.json()}
function notice(text){$('notice').textContent=text||''}
function videoName(v){if(v.device==='gopro')return 'Head-mounted GoPro';if(v.device==='showee_head')return {camera:'Head-mounted Showee · RGB',slam0:'Head-mounted Showee · left',slam1:'Head-mounted Showee · right'}[v.stream]||v.stream;if(v.device.includes('wrist'))return v.device.includes('left')?'Left wrist camera':'Right wrist camera';if(v.device.startsWith('zed'))return 'ZED · RGB';return v.device+' / '+v.stream}
function headVideo(){return UI.metadata.videos.find(v=>v.device==='showee_head'&&v.stream==='camera')||UI.metadata.videos.find(v=>v.device==='gopro')||UI.metadata.videos[0]}
function depthDevices(){return Object.entries(UI.metadata.session.cameras||{}).filter(([,camera])=>camera?.depth?.path).map(([device])=>device)}
function depthName(device){if(device==='showee_head')return 'Head-mounted Showee';if(device.startsWith('zed_'))return 'ZED · '+device.slice(4);if(device.startsWith('realsense_'))return 'RealSense · '+device.slice(10);return device}
function addDepth(device){let current=UI.panels.find(p=>p.type==='depth');if(current){current.node.scrollIntoView({block:'nearest'});return current}let devices=depthDevices();if(!devices.length)return;return addVideo({device:devices.includes(device)?device:devices[0],stream:''},true)}
function sessionList(){
    const query=$('search').value.trim().toLowerCase(),host=$('sessionList'),scroll=host.scrollTop;
    const rows=[...UI.catalog].sort(compareSessions).filter(row=>
        [row.key,row.subject,recordingDate(row),dateLabel(row),TASKS[taskType(row)]].join(' ').toLowerCase().includes(query));
    const dates=new Map();
    for(const row of rows){const day=recordingDate(row);if(!dates.has(day))dates.set(day,[]);dates.get(day).push(row)}
    host.replaceChildren();
    for(const [day,recordings] of dates){
        const heading=element('div','group-heading date-heading');
        heading.append(element('span','',day),element('span','',String(recordings.length)));host.append(heading);
        for(const [task,label] of Object.entries(TASKS)){
            const matching=recordings.filter(row=>taskType(row)===task);
            if(!matching.length)continue;
            const category=element('div','group-heading task-heading');
            category.append(element('span','',label),element('span','',String(matching.length)));host.append(category);
            for(const row of matching){
                const subject=row.subject||'Unknown participant';
                const button=element('button','session-item'+(UI.selected===row.key?' active':''));
                button.title=row.session_id;
                button.setAttribute('aria-label',day+' '+label+' '+subject+' '+row.session_id);
                button.setAttribute('aria-current',UI.selected===row.key?'true':'false');
                button.append(element('div','date',dateLabel(row)));
                const meta=element('div','meta');
                meta.append(element('span','',subject),element('span','',formatTime(row.duration_s)+(row.remote?' · remote':'')));button.append(meta);
                button.onclick=()=>selectSession(row.key);host.append(button);
            }
        }
    }
    if(!rows.length)host.append(element('div','empty','No matching sessions'));
    host.scrollTop=scroll;$('libraryCount').textContent=UI.catalog.length+' sessions';
}
function setTime(value){if(!UI.metadata)return;UI.time=Math.max(0,Math.min(UI.metadata.duration_s,Number(value)||0));$('seek').value=UI.time;$('time').value=UI.time.toFixed(2);$('clock').textContent=formatTime(UI.time)+' / '+formatTime(UI.metadata.duration_s);clearTimeout(UI.seekTimer);let now=performance.now();if(!UI.lastSeek||now-UI.lastSeek>120){UI.lastSeek=now;refresh(true)}else UI.seekTimer=setTimeout(()=>{UI.lastSeek=performance.now();refresh(true)},60)}
function stop(){UI.playing=false;$('play').textContent='▶';$('play').setAttribute('aria-label','Play')}
function disposePanels(){for(let p of UI.panels){p.abort?.abort();p.bitmap?.close();clearTimeout(p.windowTimer);clearTimeout(p.cropRetry);visibility.unobserve(p.node)}UI.panels=[];$('workspace').replaceChildren();UI.state=null}
async function selectSession(key){if(UI.loading&&key===UI.selected)return;let previous=UI.panels.map(p=>({type:p.type,device:p.device,stream:p.stream,hand:p.hand,layers:[...(p.overlays||p.layers||[])],field:p.field,finger:p.finger,windowS:p.windowS}));let token=++UI.token;UI.selected=key;UI.loading=true;stop();disposePanels();UI.metadata=null;$('phaseSummary').textContent='';notice('');sessionList();$('sessionTitle').textContent='Loading session…';$('workspace').append(element('div','empty-workspace','Loading data and available modalities…'));try{let metadata=await api('/api/session',{session:key});if(token!==UI.token)return;UI.metadata=metadata;UI.time=Math.max(0,Math.min(metadata.duration_s,headVideo()?.first_frame_time_s||0));$('seek').max=metadata.duration_s;$('time').max=metadata.duration_s;$('seek').value=UI.time;$('time').value=UI.time.toFixed(2);let row=UI.catalog.find(r=>r.key===key);$('source').textContent=recordingDate(row)+' · '+TASKS[taskType(row)]+(row.remote?' · remote':'');$('sessionTitle').textContent=dateLabel(row)+' / '+(metadata.session.subject||'Unknown participant');$('sessionMeta').textContent=metadata.session.session_id+' · '+formatTime(metadata.duration_s)+' · '+metadata.videos.length+' video streams';const handGaps=[...new Set(Object.values(metadata.session.hands||{}).map(h=>h?.keypoint_interpolation?.maximum_endpoint_gap_us).filter(Number.isFinite))];if(handGaps.length===1)$('sessionMeta').textContent+=' · hand keypoint interpolation '+(handGaps[0]/1000)+' ms';if(Object.values(metadata.session.cameras||{}).some(c=>Object.values(c?.streams||{}).some(st=>st?.external_pose_processing?.user_visually_approved)))$('sessionMeta').textContent+=' · extrinsics: reviewed motion / stationary means';if(Object.values(metadata.session.cameras||{}).some(c=>Object.values(c?.streams||{}).some(st=>st?.external_pose_processing?.frozen_template_marker_ids_used?.length===3)))$('sessionMeta').textContent+=' · three-point fit correction';if(Object.values(metadata.session.cameras||{}).some(c=>c?.approved_manual_extrinsics))$('sessionMeta').textContent+=' · manual extrinsics reviewed';$('metadataText').textContent=JSON.stringify({session:metadata.session,calibrations:metadata.calibrations},null,2);UI.loading=false;if(previous.length)restoreViews(previous);else preset('default')}catch(e){if(token!==UI.token)return;UI.loading=false;notice(e.message);$('workspace').replaceChildren(element('div','empty-workspace','Could not load this session. Check the dataset or remote connection.'))}}
function layout(){let host=$('workspace');host.classList.toggle('multiple',UI.panels.length>1);host.classList.toggle('only-videos',UI.panels.length>1&&UI.panels.every(p=>p.type==='video'||p.type==='depth'||p.type==='crop'));if(!UI.panels.length){let empty=element('div','empty-workspace');empty.append(element('h2','','Create a view layout'),element('p','','Choose video, signals, or 3D from Add view.'));host.replaceChildren(empty)}for(let [i,p]of UI.panels.entries())p.number.textContent=String(i+1).padStart(2,'0')}
function restoreViews(specs){$('workspace').replaceChildren();clearPreset();for(let spec of specs){let p;if(spec.type==='video'){let v=UI.metadata.videos.find(v=>v.device===spec.device&&v.stream===spec.stream)||headVideo();p=addVideo(v);p.overlays=new Set(spec.layers);for(let input of p.controls.querySelectorAll('input[type=checkbox]')){let key=Object.keys(OVERLAYS).find(k=>OVERLAYS[k]===input.getAttribute('aria-label'));input.checked=p.overlays.has(key)}}else if(spec.type==='crop'){p=addCrop(UI.metadata.videos.find(v=>v.device===spec.device&&v.stream===spec.stream)||headVideo());p.overlays=new Set(spec.layers||[]);for(let input of p.controls.querySelectorAll('input[type=checkbox]'))input.checked=p.overlays.has(input.dataset.layer)}else if(spec.type==='depth'){p=addDepth(spec.device)}else if(SIGNALS[spec.type]){p=addSignal(spec.type);p.field=spec.field;p.finger=spec.finger;p.setWindow(spec.windowS||2,false);let emg=p.controls.querySelector('[aria-label="EMG type"]');if(emg)emg.value=p.field;let finger=p.controls.querySelector('[aria-label="Finger group"]');if(finger)finger.value=p.finger||'all'}else if(spec.type==='scene'){addScene();p=UI.panels.at(-1)}else if(spec.type==='info')addInfo();if(p){p.hand=spec.hand||'both';let control=p.controls.querySelector('[aria-label="Hand selection"]');if(control)control.value=p.hand;p.lastRequest=0}}layout();refresh(true)}
function clearPreset(){for(let b of document.querySelectorAll('[data-preset]'))b.classList.remove('active')}
function removePanel(panel){clearTimeout(panel.cropRetry);panel.abort?.abort();panel.bitmap?.close();UI.panels=UI.panels.filter(p=>p!==panel);visibility.unobserve(panel.node);panel.node.remove();layout();clearPreset()}
function newPanel(type,title){let empty=$('workspace').querySelector('.empty-workspace');empty?.remove();let p={id:UI.nextId++,type,hand:'both',overlays:new Set(['mano','umetrack']),busy:false,lastRequest:0};p.node=element('section','panel '+(type==='video'||type==='depth'||type==='crop'?'video':type==='scene'?'scene-panel':type==='info'?'full':'signal'));p.node.dataset.panel=type;p.visible=true;p.node.setAttribute('aria-label',title+' view '+p.id);let header=element('div','panel-head'),heading=element('div','panel-title');p.number=element('span','number','01');heading.append(p.number,element('span','',title));let tools=element('div','panel-tools'),close=element('button','icon','×');close.title='Remove view';close.setAttribute('aria-label','Remove '+title);close.onclick=()=>removePanel(p);tools.append(close);header.append(heading,tools);p.node.append(header);p.controls=element('div','panel-controls');p.node.append(p.controls);UI.panels.push(p);$('workspace').append(p.node);visibility.observe(p.node);return p}
function selectControl(options,value,label,change){let select=element('select');select.setAttribute('aria-label',label);for(let [id,name]of options){let o=element('option','',name);o.value=id;select.append(o)}select.value=value;select.onchange=()=>change(select.value);return select}
function handControl(p,change){return selectControl([['both','Both hands'],['left','Left only'],['right','Right only']],p.hand,'Hand selection',value=>{p.hand=value;change()})}
function addVideo(v=headVideo(),depth=false){if(!v)return;let p=newPanel(depth?'depth':'video',depth?'Depth':'Video');p.device=v.device;p.stream=v.stream;p.canvas=element('canvas');p.stage=element('div','video-stage');p.placeholder=element('div','video-placeholder','Reading the timeline…');p.stage.append(p.canvas,p.placeholder);p.node.append(p.stage);p.caption=element('div','video-caption');p.frameInfo=element('span','','');p.warning=element('span','video-warning','');p.caption.append(p.frameInfo,p.warning);p.node.append(p.caption);
if(!depth){p.controls.append(selectControl(UI.metadata.videos.map(x=>[x.device+'/'+x.stream,videoName(x)]),p.device+'/'+p.stream,'Video camera',value=>{[p.device,p.stream]=value.split('/');p.lastRequest=0;p.bitmap?.close();p.bitmap=null;p.placeholder.hidden=false;p.placeholder.textContent='Loading camera…';refresh(true)}));p.controls.append(handControl(p,()=>{p.lastRequest=0;refresh(true)}));let overlays=element('div','overlay-controls');for(let [name,label]of Object.entries(OVERLAYS)){let wrapper=element('label'),input=element('input');if(name==='umetrack')wrapper.title=Object.values(UI.metadata.session.hands).some(h=>h?.umetrack_fit)?'Fitted UmeTrack labels · default hand shape':'UmeTrack hand mesh projection';input.type='checkbox';input.checked=p.overlays.has(name);input.setAttribute('aria-label',label);input.onchange=()=>{input.checked?p.overlays.add(name):p.overlays.delete(name);p.lastRequest=0;refresh(true)};wrapper.append(input,document.createTextNode(label));overlays.append(wrapper)}p.controls.append(overlays)}else{let devices=depthDevices();p.controls.append(selectControl(devices.map(device=>[device,depthName(device)]),p.device,'Depth camera',device=>{p.device=device;p.lastRequest=0;p.bitmap?.close();p.bitmap=null;p.placeholder.hidden=false;p.placeholder.textContent='Loading depth…';p.warning.textContent='';refresh(true)}));p.controls.append(element('span','control-label','0–3 m · black pixels indicate invalid depth'));p.overlays.clear()}layout();refresh(true);return p}

function addCrop(v=headVideo()){
    if(!v)return;
    let p=newPanel('crop','Hand crop · original handedness');p.device=v.device;p.stream=v.stream;p.cards={};p.overlays.clear();
    p.controls.append(selectControl(UI.metadata.videos.map(x=>[x.device+'/'+x.stream,videoName(x)]),p.device+'/'+p.stream,'Crop camera',value=>{[p.device,p.stream]=value.split('/');p.lastRequest=0;p.frameKey=null;refresh(true)}));
    p.controls.append(handControl(p,()=>{p.frameKey=null;p.lastRequest=0;refresh(true)}));
    p.controls.append(element('span','control-label','256 × 256 · no mirroring · original handedness'));
    const overlays=element('div','overlay-controls');
    for(let name of ['mano','umetrack','markers','phase']){
        let label=element('label'),input=element('input');input.type='checkbox';input.dataset.layer=name;
        input.setAttribute('aria-label',OVERLAYS[name]);input.onchange=()=>{input.checked?p.overlays.add(name):p.overlays.delete(name);p.frameKey=null;refresh(true)};
        label.append(input,document.createTextNode(OVERLAYS[name]));overlays.append(label);
    }
    p.controls.append(overlays);
    p.grid=element('div','crop-grid');
    for(let hand of ['left','right']){let card=element('div','crop-card'),label=element('strong','',hand==='left'?'Left hand':'Right hand'),canvas=element('canvas'),status=element('div','crop-status','Loading…');canvas.width=256;canvas.height=256;card.append(label,canvas,status);p.grid.append(card);p.cards[hand]={card,canvas,status}}
    p.caption=element('div','video-caption','Uses the source video frame index; blank means no valid crop');p.node.append(p.grid,p.caption);layout();refresh(true);return p;
}
async function cropFrame(p,force=false){
    if(UI.loading||document.hidden||p.visible===false||!UI.metadata||!UI.panels.includes(p))return;
    if(p.busy){if(force)p.pending=true;return}if(!force&&Date.now()-p.lastRequest<120)return;
    p.busy=true;p.lastRequest=Date.now();let token=UI.token,time=UI.time,settings=cropSettings(p),controller=new AbortController();p.abort=controller;
    try{
        const args={session:UI.selected,device:p.device,stream:p.stream,time_s:time};
        const meta=await api('/api/hand-crops',args,controller.signal),key=settings+'/'+meta.frame_index+'/'+meta.crop_id;
        if(token!==UI.token||settings!==cropSettings(p)||!UI.panels.includes(p))return;
        for(let hand of ['left','right'])p.cards[hand].card.hidden=p.hand!=='both'&&p.hand!==hand;
        clearTimeout(p.windowTimer);clearTimeout(p.cropRetry);if(meta.status==='generating')p.cropRetry=setTimeout(()=>cropFrame(p,true),5000)
        if(p.frameKey===key&&!force)return;
        const selected=['left','right'].filter(h=>p.hand==='both'||p.hand===h);
        // Explicit camera row prevents nearest-timestamp ties from selecting another frame.
        const overlayPromise=p.overlays.size?api('/api/overlay',{...args,frame_index:meta.frame_index,crop_id:meta.crop_id,layers:[...p.overlays].join(','),hand:p.hand,include_faces:(p.overlays.has('mano')&&!UI.manoFaces)||(p.overlays.has('umetrack')&&!UI.umetrackFaces)},controller.signal):Promise.resolve(null);
        const [frames,overlay]=await Promise.all([Promise.all(selected.map(async hand=>{
            let record=meta.hands[hand];if(!record.valid)return {hand,record,bitmap:null};
            let response=await fetch('/api/hand-crop?'+new URLSearchParams({...args,hand,frame_index:meta.frame_index,crop_id:meta.crop_id}),{signal:controller.signal});
            if(!response.ok)throw Error('Crop read failed: '+await response.text());
            return {hand,record,bitmap:await createImageBitmap(await response.blob())};
        })),overlayPromise]);
        if(token!==UI.token||settings!==cropSettings(p)||!UI.panels.includes(p)){for(let f of frames)f.bitmap?.close();return}
        if(overlay&&overlay.frame_index!==meta.frame_index){for(let f of frames)f.bitmap?.close();throw Error('Overlay and crop frame indices differ')}
        for(let h of Object.values(overlay?.hands||{})){if(h.mano?.faces)UI.manoFaces=h.mano.faces;if(h.umetrack?.faces)UI.umetrackFaces=h.umetrack.faces}
        for(let {hand,record,bitmap}of frames){
            let card=p.cards[hand],g=card.canvas.getContext('2d');g.clearRect(0,0,256,256);
            if(bitmap){
                g.drawImage(bitmap,0,0);bitmap.close();
                if(overlay)drawOverlay(g,card.canvas,cropOverlay(overlay,hand,record.source_to_crop),p.overlays);
                card.status.textContent='bbox [ '+record.bbox.map(v=>v.toFixed(1)).join(', ')+' ] px';
                const missing=[...p.overlays].filter(k=>k!=='phase'&&!overlay?.hands[hand]?.[k]);
                if(missing.length)card.status.textContent+=' · '+missing.map(k=>OVERLAYS[k]).join(' / ')+' annotations unavailable';
            }else card.status.textContent=({crop_not_generated:'Crop not generated',source_frame_or_time_unavailable:'Source frame or timestamp unavailable',camera_pose_unavailable:'Camera pose unavailable',insufficient_points:'Insufficient valid hand points',crop_range_invalid:'Invalid crop bounds',image_generation_failed:'Image generation failed',not_processed:'Not processed'})[record.reason]||record.reason;
            card.status.title=JSON.stringify(record,null,2);
        }
        const dt=(time-meta.frame_time_s)*1000;
        for(let card of Object.values(p.cards))card.canvas.style.opacity=Math.abs(dt)>200?'.4':'1';
        p.caption.textContent=videoName({device:p.device,stream:p.stream})+' · frame '+meta.frame_index+' · Δt '+dt.toFixed(1)+' ms'+(Math.abs(dt)>200?' · no synchronized video; showing a nearby crop':'')+(meta.status==='partial'?' · generating in batches':'')+(Object.values(meta.hands).some(h=>h.geometry_quality?.includes('historical'))?' · historical projection; extrinsic provenance incomplete':'');
        if(meta.status==='generating'){let progress=meta.generation_progress;p.caption.textContent='Generating hand crops · '+(progress.frames_done/progress.frames_total*100).toFixed(1)+'% · '+progress.images.toLocaleString()+' images; loads automatically when complete'}
        if(overlay?.warnings?.length)p.caption.textContent+=' · '+[...new Set(overlay.warnings)].join('; ');
        if(p.lastCropId!==undefined&&p.lastCropId!==meta.crop_id)for(let sibling of UI.panels)if(sibling.type==='video'&&sibling.device===p.device&&sibling.stream===p.stream&&sibling.overlays.has('bbox'))videoFrame(sibling,true);
        p.lastCropId=meta.crop_id;p.frameKey=key;
    }catch(e){if(e.name!=='AbortError'&&token===UI.token&&UI.panels.includes(p)){p.caption.textContent=e.message;for(let h of Object.values(p.cards)){h.canvas.getContext('2d').clearRect(0,0,256,256);h.status.textContent='Read unavailable'}}}
    finally{p.busy=false;if(token===UI.token&&UI.panels.includes(p)&&(p.pending||(!UI.playing&&Math.abs(UI.time-time)>.05)||settings!==cropSettings(p))){p.pending=false;cropFrame(p,true)}}
}

function addTimeWindowControl(p){
    p.windowS=2;
    const control=element('div','time-window-control'),label=element('span','','Time window'),slider=element('input'),number=element('input'),hint=element('span','time-window-hint');
    slider.type='range';slider.min='.2';slider.max='60';slider.step='.2';slider.setAttribute('aria-label','Time window (seconds)');
    number.type='number';number.min='.2';number.max='60';number.step='.2';number.setAttribute('aria-label','Time window value (seconds)');
    p.setWindow=(value,refreshState=true)=>{
        value=Number(value);if(!Number.isFinite(value))return;
        p.windowS=Math.min(60,Math.max(.2,Math.round(value*5)/5));
        slider.value=p.windowS;number.value=p.windowS.toFixed(1);hint.textContent='s · centered ±'+Number((p.windowS/2).toFixed(1))+' s';
        slider.setAttribute('aria-valuetext',p.windowS.toFixed(1)+' s; each side spans '+(p.windowS/2).toFixed(1)+' s');
        if(refreshState){clearTimeout(p.windowTimer);p.windowTimer=setTimeout(()=>{if(UI.panels.includes(p))requestState(true)},100)}
    };
    slider.oninput=()=>p.setWindow(slider.value);number.onchange=()=>p.setWindow(number.value);number.onkeydown=e=>{if(e.key==='Enter')p.setWindow(number.value)};
    control.append(label,element('span','time-window-limit','0.2 s'),slider,element('span','time-window-limit','60 s'),number,hint);p.controls.append(control);p.setWindow(2,false);
}
function addSignal(type){let p=newPanel(type,SIGNALS[type].name);p.field=SIGNALS[type].field;addTimeWindowControl(p);p.body=element('div','signal-body');p.node.append(p.body);p.node.append(element('div','signal-foot','Independent channel scales · cursor marks current time · long windows are subsampled'));p.controls.append(handControl(p,()=>drawSignal(p)));if(type==='emg')p.controls.append(selectControl([['emg_filtered',Object.values(UI.metadata.session.hands).some(h=>h?.filter?.implementation==='session_fft_zero_phase')?'Filtered EMG · session FFT':'Filtered EMG'],['emg_raw','Raw EMG']],p.field,'EMG type',v=>{p.field=v;requestState(true)}));if(type==='joint')p.controls.append(selectControl([['all','All 20 channels'],['0','Thumb'],['1','Index'],['2','Middle'],['3','Ring'],['4','Little']],'all','Finger group',v=>{p.finger=v;drawSignal(p)}));layout();refresh(true);return p}
function addScene(){let p=newPanel('scene','3D · hands and world frame');p.layers=new Set(['mano','umetrack','markers']);for(let [name,label]of [['mano','MANO'],['umetrack','UmeTrack'],['markers','Mocap points'],['cameras','Cameras']]){let wrapper=element('label','control-label'),input=element('input');input.type='checkbox';input.checked=p.layers.has(name);input.onchange=()=>{input.checked?p.layers.add(name):p.layers.delete(name);drawScene(p)};wrapper.append(input,document.createTextNode(' '+label));p.controls.append(wrapper)}let fit=element('button','','Fit view');fit.onclick=()=>{p.center=null;p.zoom=1;drawScene(p)};p.controls.append(fit);p.canvas=element('canvas','scene');p.node.append(p.canvas,element('div','scene-legend','Drag to rotate · scroll to zoom · teal left / orange right · UmeTrack mesh'));p.yaw=.6;p.pitch=-.3;p.zoom=1;p.center=null;let drag=null;p.canvas.onpointerdown=e=>{drag=[e.clientX,e.clientY];p.canvas.setPointerCapture(e.pointerId)};p.canvas.onpointermove=e=>{if(drag){p.yaw+=(e.clientX-drag[0])*.008;p.pitch+=(e.clientY-drag[1])*.008;drag=[e.clientX,e.clientY];drawScene(p)}};p.canvas.onpointerup=()=>drag=null;p.canvas.onwheel=e=>{e.preventDefault();p.zoom*=Math.exp(-e.deltaY*.001);drawScene(p)};layout();refresh(true)}
function addInfo(){let p=newPanel('info','Phase and label status');p.controls.remove();p.body=element('div','metadata-table');p.node.append(p.body);layout();refresh(true)}
function preset(name){if(!UI.metadata||UI.loading)return;disposePanels();for(let b of document.querySelectorAll('[data-preset]'))b.classList.toggle('active',b.dataset.preset===name);addVideo();if(name==='crop'){UI.panels[0].overlays=new Set(['bbox']);for(let input of UI.panels[0].controls.querySelectorAll('input[type=checkbox]'))input.checked=input.getAttribute('aria-label')==='Hand crop boxes';addCrop()}if(name==='emg')addSignal('emg');if(name==='wrist')addSignal('wrist');if(name==='compare'){let external=UI.metadata.videos.filter(v=>/^(zed_|realsense_)/.test(v.device)&&v.stream==='rgb');for(let v of (external.length?external:UI.metadata.videos.filter(v=>v!==headVideo())).slice(0,3))addVideo(v)}layout();refresh(true)}
function viewChoices(){let host=$('viewChoices');host.replaceChildren();function choice(name,description,callback){let b=element('button','view-choice');b.append(element('strong','',name),element('small','',description));b.onclick=()=>{$('viewDialog').close();clearPreset();callback()};host.append(b)}choice('Video','Add a camera and select its overlays',()=>addVideo());choice('Hand crop','Original handedness · JPEG + offset · source-video bounding boxes',()=>addCrop());for(let [name,info]of Object.entries(SIGNALS))choice(info.name,info.description,()=>addSignal(name));choice('3D world view','Mocap points, hand models, and camera poses',addScene);if(depthDevices().length)choice('Depth','Switch cameras · native depth timeline · metres',()=>addDepth());choice('Phase and label status','Task phases, label validity, and missing data',addInfo);$('viewDialog').showModal()}
function goodPoint(layer,i){let uv=layer.uv[i];return layer.depth[i]>.01&&uv&&uv.every(Number.isFinite)}
function cropSettings(p){return p.device+'/'+p.stream+'/'+p.hand+'/'+[...p.overlays].sort().join(',')}
function cropOverlay(overlay,hand,affine){
    const source=overlay.hands[hand]||{},target={};
    for(let kind of ['mano','umetrack','markers']){
        let layer=source[kind];if(!layer)continue;
        target[kind]={...layer,uv:layer.uv.map(([x,y])=>{
            let w=affine[2][0]*x+affine[2][1]*y+affine[2][2];
            return [(affine[0][0]*x+affine[0][1]*y+affine[0][2])/w,(affine[1][0]*x+affine[1][1]*y+affine[1][2])/w];
        })};
    }
    return {...overlay,image_size:[256,256],hands:{[hand]:target}};
}
function drawVideo(p){if(!p.bitmap)return;let c=p.canvas,g=c.getContext('2d');if(c.width!==p.bitmap.width||c.height!==p.bitmap.height){c.width=p.bitmap.width;c.height=p.bitmap.height}g.drawImage(p.bitmap,0,0);if(p.type!=='depth')drawOverlay(g,c,p.overlay,p.overlays)}
function drawOverlay(g,c,o,layers){if(!o)return;let sx=c.width/o.image_size[0],sy=c.height/o.image_size[1];let faces=[];for(let [hand,h]of Object.entries(o.hands)){for(let kind of ['mano','umetrack']){let mesh=h[kind];if(!mesh||!layers.has(kind))continue;for(let f of (mesh.faces||(kind==='mano'?UI.manoFaces:UI.umetrackFaces)||[])){if(!f.every(i=>goodPoint(mesh,i)))continue;let uv=f.map(i=>[mesh.uv[i][0]*sx,mesh.uv[i][1]*sy]);if(uv.every(v=>v[0]<0)||uv.every(v=>v[0]>c.width)||uv.every(v=>v[1]<0)||uv.every(v=>v[1]>c.height))continue;faces.push({uv,z:f.reduce((v,i)=>v+mesh.depth[i],0),color:kind==='umetrack'?(hand==='left'?'rgba(117,174,246,.42)':'rgba(190,144,230,.42)'):(hand==='left'?'rgba(41,188,166,.32)':'rgba(242,174,100,.32)')})}}}
faces.sort((a,b)=>b.z-a.z);for(let f of faces){g.beginPath();g.moveTo(...f.uv[0]);g.lineTo(...f.uv[1]);g.lineTo(...f.uv[2]);g.closePath();g.fillStyle=f.color;g.fill()}
for(let [hand,h]of Object.entries(o.hands)){if(h.markers&&layers.has('markers')){let markers=h.markers;for(let i=0;i<markers.uv.length;i++)if(markers.valid[i]&&goodPoint(markers,i)){g.beginPath();g.arc(markers.uv[i][0]*sx,markers.uv[i][1]*sy,3,0,Math.PI*2);g.fillStyle=COLORS[hand];g.fill();g.strokeStyle='#fff';g.lineWidth=.6;g.stroke()}}}
if(layers.has('bbox'))for(let [hand,h]of Object.entries(o.hands)){let box=h.crop?.bbox;if(!h.crop?.valid||!box)continue;g.strokeStyle=COLORS[hand];g.lineWidth=2;g.strokeRect(box[0]*sx,box[1]*sy,(box[2]-box[0])*sx,(box[3]-box[1])*sy);g.font='13px system-ui';g.fillStyle=COLORS[hand];g.fillText(hand==='left'?'Left crop':'Right crop',box[0]*sx+5,Math.max(15,box[1]*sy-5))}
if(layers.has('phase')&&o.annotations.length){g.font='14px system-ui';let lines=o.annotations;let width=Math.min(c.width-24,Math.max(...lines.map(x=>g.measureText(x).width))+24);g.fillStyle='rgba(15,25,28,.78)';g.fillRect(12,12,width,12+lines.length*23);g.fillStyle='#fff';lines.forEach((x,i)=>g.fillText(x,24,34+i*23,width-24))}}
async function videoFrame(p,force=false){if(UI.loading||document.hidden||p.visible===false||!UI.metadata||!UI.panels.includes(p))return;if(p.busy){if(force)p.pending=true;return;}let interval=UI.panels.filter(v=>v.visible!==false&&(v.type==='video'||v.type==='depth')).length<=1?50:100;if(!force&&Date.now()-p.lastRequest<interval)return;p.busy=true;p.lastRequest=Date.now();let token=UI.token,time=UI.time,settings=p.device+p.stream+p.hand+[...p.overlays].join(','),controller=new AbortController();p.abort=controller;try{let args={session:UI.selected,device:p.device,stream:p.stream,time_s:time},framePath=p.type==='depth'?'/api/depth':'/api/frame';let [response,overlay]=await Promise.all([fetch(framePath+'?'+new URLSearchParams(args),{signal:controller.signal}),p.type==='video'&&p.overlays.size?api('/api/overlay',{...args,layers:[...p.overlays].join(','),hand:p.hand,include_faces:(p.overlays.has('mano')&&!UI.manoFaces)||(p.overlays.has('umetrack')&&!UI.umetrackFaces)},controller.signal):Promise.resolve(null)]);if(!response.ok)throw Error('No readable frame at this time');let bitmap=await createImageBitmap(await response.blob());if(token!==UI.token||!UI.panels.includes(p)||settings!==p.device+p.stream+p.hand+[...p.overlays].join(',')){bitmap.close();return}p.bitmap?.close();p.bitmap=bitmap;p.overlay=overlay;for(let h of Object.values(overlay?.hands||{})){if(h.mano?.faces)UI.manoFaces=h.mano.faces;if(h.umetrack?.faces)UI.umetrackFaces=h.umetrack.faces}p.placeholder.hidden=true;drawVideo(p);let frame=response.headers.get('X-Frame-Index')||response.headers.get('X-Depth-Frame')||overlay?.frame_index||'—';let dt=overlay?((time-overlay.frame_time_s)*1000):null;p.frameInfo.textContent=(p.type==='depth'?'Depth':videoName({device:p.device,stream:p.stream}))+' · frame '+frame+(dt!==null?' · Δt '+dt.toFixed(1)+' ms':'');let warnings=overlay?.warnings||[];if(dt!==null&&Math.abs(dt)>200)warnings=['No synchronized video; showing a nearby frame',...warnings];p.warning.textContent=[...new Set(warnings)].join('; ');p.warning.title=p.warning.textContent;if(dt!==null&&Math.abs(dt)>200)p.canvas.style.opacity='.4';else p.canvas.style.opacity='1'}catch(e){if(e.name!=='AbortError'&&token===UI.token&&UI.panels.includes(p)){p.warning.textContent=e.message;if(!p.bitmap)p.placeholder.textContent=e.message}}finally{p.busy=false;if(token===UI.token&&UI.panels.includes(p)&&(p.pending||(!UI.playing&&Math.abs(UI.time-time)>.05)||settings!==p.device+p.stream+p.hand+[...p.overlays].join(','))){p.pending=false;videoFrame(p,true)}}}
function plot(canvas,rows,times,unit,channels=null,field='',windowS=2){
    const width=canvas.clientWidth||400,allN=Array.isArray(rows[0])?rows[0].length:1;
    const n=channels?.length||allN,band=field==='wrist_angles'?76:52,height=n*band+24;
    canvas.style.height=height+'px';
    const pw=Math.round(width*devicePixelRatio),ph=Math.round(height*devicePixelRatio);
    if(canvas.width!==pw||canvas.height!==ph){canvas.width=pw;canvas.height=ph}
    const g=canvas.getContext('2d'),margin=100,plotW=Math.max(1,width-margin-8);
    g.setTransform(devicePixelRatio,0,0,devicePixelRatio,0,0);g.clearRect(0,0,width,height);g.font='10px system-ui';
    if(!rows.length){g.fillStyle='#94a1a8';g.fillText('No valid samples in this time window',12,30);return}
    const colors=['#419c8c','#6785bf','#cf9c5f','#bc7596','#9784b6','#8b9b64','#67a3b5','#af9583'];
    const isIMU=field==='imu'||field.endsWith('_imu_aligned');
    function tick(v){if(v===0)return '0';if(Math.abs(v)>=1e4||Math.abs(v)<.001)return v.toExponential(1);return Number(v.toPrecision(3)).toString()}
    for(let ch=0;ch<n;ch++){
        const actual=channels?.[ch]??ch,values=rows.map(r=>Array.isArray(r)?r[actual]:r);
        const finite=values.filter(v=>v!==null&&Number.isFinite(v));
        let lo=finite.length?Math.min(...finite):0,hi=finite.length?Math.max(...finite):1;
        const pad=hi===lo?Math.max(Math.abs(lo)*.05,.01):(hi-lo)*.05;lo-=pad;hi+=pad;
        const top=ch*band,yTop=top+10,yBottom=top+band-12;
        const label=field==='wrist_angles'?['pitch','yaw'][actual]:isIMU?['ax','ay','az','gx','gy','gz'][actual]:'C'+actual;
        const channelUnit=isIMU?(actual<3?'m/s²':'rad/s'):unit;
        g.fillStyle='#60727b';g.textAlign='left';g.fillText(label||String(actual),1,top+15);g.fillText(channelUnit,1,top+29);
        g.textAlign='right';
        for(const f of [0,.5,1]){
            const y=yTop+(yBottom-yTop)*f,value=hi-(hi-lo)*f;
            g.fillStyle='#809099';g.fillText(finite.length?tick(value):'—',margin-7,y+3);
            g.strokeStyle='#eef2f4';g.lineWidth=1;g.beginPath();g.moveTo(margin,y);g.lineTo(margin+plotW,y);g.stroke();
        }
        g.strokeStyle='#d7e1e5';g.beginPath();g.moveTo(margin,yTop);g.lineTo(margin,yBottom);g.stroke();
        g.strokeStyle=colors[ch%8];g.lineWidth=1.1;g.beginPath();let started=false;
        for(let i=0;i<values.length;i++){
            if(values[i]===null||!Number.isFinite(values[i])){started=false;continue}
            const x=margin+(times[i]/windowS+.5)*plotW,y=yBottom-(values[i]-lo)/(hi-lo)*(yBottom-yTop);
            if(started)g.lineTo(x,y);else{g.moveTo(x,y);started=true}
        }
        g.stroke();
    }
    g.strokeStyle='#b4c8c0';g.beginPath();g.moveTo(margin+plotW/2,0);g.lineTo(margin+plotW/2,height-24);g.stroke();
    g.fillStyle='#809099';
    for(let i=0;i<=4;i++){
        const value=(i/4-.5)*windowS,label=i===2?'0 s · current':(value>0?'+':'')+Number(value.toFixed(2))+' s';
        g.textAlign=i===0?'left':i===4?'right':'center';g.fillText(label,margin+i/4*plotW,height-4);
    }
    g.textAlign='left';
}
function drawSignal(p){
    const state=p.signalState;if(!state||p.loadedWindowS!==p.windowS)return;
    p.charts??=new Map();let active=new Set(),found=false;
    function chart(title,rows,times,unit,field,channels,current,age,timestamp){
        found=true;const key=title+'|'+field;active.add(key);let chart=p.charts.get(key);
        if(!chart){
            p.body.querySelector('.empty')?.remove();
            const row=element('div','signal-row'),heading=element('h3');
            heading.append(element('span','',title),element('span','',unit));
            const status=element('div','current-sample'),values=element('div','current-values'),canvas=element('canvas');
            row.append(heading,status,values,canvas);p.body.append(row);
            chart={row,canvas,status,values,cells:new Map()};p.charts.set(key,chart);
        }
        chart.status.textContent='Current sample · age '+(age/1000).toFixed(2)+' ms';
        chart.status.title='Original sample timestamp: '+timestamp+' µs';
        const n=Array.isArray(current)?current.length:(Array.isArray(rows[0])?rows[0].length:1);
        const selected=channels||Array.from({length:n},(_,i)=>i),keep=new Set(selected);
        for(const ch of selected){
            const isIMU=field==='imu'||field.endsWith('_imu_aligned');
            const label=field==='wrist_angles'?['pitch','yaw'][ch]:isIMU?['ax','ay','az','gx','gy','gz'][ch]:'C'+ch;
            const channelUnit=isIMU?(ch<3?'m/s²':'rad/s'):unit;
            let cell=chart.cells.get(ch);
            if(!cell){const box=element('div','current-value'),name=element('span','channel-name',label),value=element('strong'),suffix=element('span','channel-unit',channelUnit);box.append(name,value,suffix);chart.values.append(box);cell={box,value};chart.cells.set(ch,cell)}
            const value=Array.isArray(current)?current[ch]:current,valid=value!==null&&Number.isFinite(value);
            cell.value.textContent=valid?Number(value.toPrecision(6)).toString():'—';
            cell.box.title=valid?'Original value: '+String(value)+' '+channelUnit:'Invalid sample or no synchronized data';
            cell.box.classList.toggle('invalid',!valid);
        }
        for(const [ch,cell] of chart.cells)if(!keep.has(ch)){cell.box.remove();chart.cells.delete(ch)}
        plot(chart.canvas,rows,times,unit,channels,field,p.windowS);
    }
    for(const [hand,h] of Object.entries(state.hands)){
        if(p.hand!=='both'&&p.hand!==hand)continue;
        const fields=p.type==='cameraIMU'?Object.keys(h.traces).filter(k=>k.endsWith('_imu_aligned')||k==='camera_imu_aligned'):[p.field];
        for(const field of fields){
            const rows=h.traces[field];if(!rows)continue;
            const channels=p.type==='joint'&&p.finger&&p.finger!=='all'?Array.from({length:4},(_,i)=>Number(p.finger)*4+i):null;
            chart((hand==='left'?'Left hand':'Right hand')+(p.type==='cameraIMU'?' · '+field.replace('_imu_aligned',''):''),rows,h.time_s,SIGNALS[p.type].unit,field,channels,h.instant[field],h.sample_age_us,h.timestamp_us);
        }
    }
    if(p.type==='cameraIMU')for(const [device,v]of Object.entries(state.camera_imu||{}))chart(device+' · native',v.samples,v.time_s,SIGNALS[p.type].unit,'imu',null,v.instant,v.sample_age_us,v.timestamp_us);
    for(const [key,chart]of p.charts)if(!active.has(key)){chart.row.remove();p.charts.delete(key)}
    if(!found&&!p.body.querySelector('.empty'))p.body.append(element('div','empty','This session does not contain this modality'));
}
function drawInfo(p){if(!UI.state)return;p.body.replaceChildren();let state=UI.state;let intervals=state.interactions||[];p.body.append(element('h3','','Current phase'));for(let i of intervals)p.body.append(element('div','',i.phase+' · '+i.object_instances.join(' / ')+' · '+i.label_kind));if(!intervals.length)p.body.append(element('div','','No interaction-phase labels at this time'));p.body.append(element('h3','','Label validity'));for(let [hand,h]of Object.entries(state.hands))p.body.append(element('div','',(hand==='left'?'Left hand':'Right hand')+' · original row '+h.row+' · MANO '+(h.native_generated_validity?.mano??h.instant.mano_valid??h.instant.label_valid??'Not recorded')+' · finger angles '+(h.native_generated_validity?.umetrack??h.instant.joint_angles_valid??h.instant.label_valid??'Not recorded')+' · sample age '+(h.sample_age_us/1000).toFixed(1)+' ms'));let details=element('details');details.append(element('summary','','Missing data and current values'));details.append(element('pre','',JSON.stringify({hands:Object.fromEntries(Object.entries(state.hands).map(([k,h])=>[k,h.instant])),missing:state.missing},null,2)));p.body.append(details)}
function drawScene(p){let state=UI.state;if(!state)return;let c=p.canvas,w=c.clientWidth||500,h=c.clientHeight||380;let pw=Math.round(w*devicePixelRatio),ph=Math.round(h*devicePixelRatio);if(c.width!==pw||c.height!==ph){c.width=pw;c.height=ph}let g=c.getContext('2d');g.setTransform(devicePixelRatio,0,0,devicePixelRatio,0,0);g.clearRect(0,0,w,h);let points=[];for(let hand of Object.values(state.hands))if(hand.keypoints)points.push(...hand.keypoints.filter((v,i)=>hand.keypoint_valid[i]));if(!points.length){g.fillStyle='#8a979e';g.fillText('No valid mocap points at this time',20,30);return}if(!p.center){let min=[0,1,2].map(k=>Math.min(...points.map(v=>v[k]))),max=[0,1,2].map(k=>Math.max(...points.map(v=>v[k])));p.center=min.map((v,k)=>(v+max[k])/2);p.extent=Math.max(...max.map((v,k)=>v-min[k]),.2)}function project(v){let [x,y,z]=v.map((v,k)=>(v-p.center[k])/p.extent),a=x*Math.cos(p.yaw)+z*Math.sin(p.yaw),b=-x*Math.sin(p.yaw)+z*Math.cos(p.yaw),d=y*Math.sin(p.pitch)+b*Math.cos(p.pitch),vertical=y*Math.cos(p.pitch)-b*Math.sin(p.pitch),scale=Math.min(w,h)*.7*p.zoom;return [w/2+a*scale,h/2-vertical*scale,d]}function line(a,b,color){a=project(a);b=project(b);g.strokeStyle=color;g.beginPath();g.moveTo(a[0],a[1]);g.lineTo(b[0],b[1]);g.stroke()}let triangles=[];for(let [hand,data]of Object.entries(state.hands)){let geometry=data.geometry;for(let kind of ['mano','umetrack']){let source=kind==='mano'?geometry?.vertices:geometry?.umetrack_vertices,faces=kind==='mano'?geometry?.faces:geometry?.umetrack_faces;if(!p.layers.has(kind)||!source||!faces)continue;let vertices=source.map(project);for(let face of faces)triangles.push({uv:face.map(i=>vertices[i]),color:kind==='umetrack'?(hand==='left'?'rgba(117,174,246,.7)':'rgba(190,144,230,.7)'):(hand==='left'?'rgba(53,181,159,.7)':'rgba(226,157,86,.7)')})}}triangles.sort((a,b)=>a.uv.reduce((s,v)=>s+v[2],0)-b.uv.reduce((s,v)=>s+v[2],0));for(let f of triangles){g.fillStyle=f.color;g.beginPath();g.moveTo(f.uv[0][0],f.uv[0][1]);g.lineTo(f.uv[1][0],f.uv[1][1]);g.lineTo(f.uv[2][0],f.uv[2][1]);g.fill()}for(let [hand,data]of Object.entries(state.hands)){if(p.layers.has('markers'))for(let i=0;i<(data.keypoints||[]).length;i++)if(data.keypoint_valid[i]){let uv=project(data.keypoints[i]);g.fillStyle=COLORS[hand];g.beginPath();g.arc(uv[0],uv[1],2.5,0,Math.PI*2);g.fill()}}if(p.layers.has('markers')&&state.mocap)for(let i=0;i<state.mocap.points.length;i++)if(state.mocap.valid[i]){let v=project(state.mocap.points[i]);g.fillStyle='#a8b6bc';g.fillRect(v[0],v[1],2,2)}if(p.layers.has('cameras'))for(let cam of state.cameras)if(cam.world_pose){let m=cam.world_pose,o=[m[0][3],m[1][3],m[2][3]],corners=[[-.04,-.03,.07],[.04,-.03,.07],[.04,.03,.07],[-.04,.03,.07]].map(v=>[0,1,2].map(k=>o[k]+m[k][0]*v[0]+m[k][1]*v[1]+m[k][2]*v[2]));for(let i=0;i<4;i++){line(o,corners[i],'#8ba1bd');line(corners[i],corners[(i+1)%4],'#8ba1bd')}}}
async function requestState(force=false){
    if(UI.stateBusy){if(force)UI.statePending=true;return}
    const visible=UI.panels.filter(p=>p.visible!==false),consumers=visible.filter(p=>SIGNALS[p.type]||p.type==='scene'||p.type==='info');
    if(UI.loading||document.hidden||!consumers.length)return;
    if(!force&&Date.now()-UI.lastState<120)return;
    UI.stateBusy=true;UI.lastState=Date.now();const token=UI.token,time=UI.time;
    try{
        const windows=[...new Set(consumers.filter(p=>SIGNALS[p.type]).map(p=>p.windowS))];if(!windows.length)windows.push(2);
        const responses=await Promise.all(windows.map(async(windowS,index)=>{
            const panels=consumers.filter(p=>SIGNALS[p.type]&&p.windowS===windowS),fields=new Set();
            for(let p of panels){
                if(p.type==='cameraIMU'){for(let h of Object.values(UI.metadata.session.hands))if(h)for(let key of Object.keys(h.arrays))if(key.endsWith('_imu_aligned')||key==='camera_imu_aligned')fields.add(key)}
                else fields.add(p.field);
            }
            const state=await api('/api/state',{session:UI.selected,time_s:time,window_s:windowS,fields:[...fields].join(','),geometry:index===0&&consumers.some(p=>p.type==='scene'),native_imu:panels.some(p=>p.type==='cameraIMU'),mocap_points:index===0&&consumers.some(p=>p.type==='scene')});
            return [windowS,state];
        }));
        if(token!==UI.token)return;
        UI.state=responses[0][1];const states=new Map(responses);
        for(let p of UI.panels){
            if(p.visible===false)continue;
            if(SIGNALS[p.type]){if(states.has(p.windowS)){p.signalState=states.get(p.windowS);p.loadedWindowS=p.windowS;drawSignal(p)}}
            else if(p.type==='scene')drawScene(p);else if(p.type==='info')drawInfo(p);
        }
        $('phaseSummary').textContent=UI.state.interactions?.length?UI.state.interactions.map(x=>({interaction:'Interaction',preparation:'Preparation'}[x.phase]||x.phase)).join(' / '):'All views share the same timeline';
    }catch(e){if(token===UI.token)notice(e.message)}
    finally{UI.stateBusy=false;if(UI.statePending||token!==UI.token||(!UI.playing&&Math.abs(UI.time-time)>.1)){UI.statePending=false;requestState(true)}}
}
function refresh(force=false){if(!UI.metadata||UI.loading)return;for(let p of UI.panels){if(p.type==='video'||p.type==='depth')videoFrame(p,force);else if(p.type==='crop')cropFrame(p,force)}requestState(force);$('clock').textContent=formatTime(UI.time)+' / '+formatTime(UI.metadata.duration_s)}
$('search').oninput=sessionList;$('addView').onclick=()=>{if(UI.metadata&&!UI.loading)viewChoices()};$('details').onclick=()=>{if(UI.metadata)$('infoDialog').showModal()};for(let b of document.querySelectorAll('.close-dialog'))b.onclick=()=>b.closest('dialog').close();for(let b of document.querySelectorAll('[data-preset]'))b.onclick=()=>preset(b.dataset.preset);$('seek').oninput=e=>setTime(e.target.value);$('time').oninput=e=>setTime(e.target.value);$('back').onclick=()=>setTime(UI.time-1);$('forward').onclick=()=>setTime(UI.time+1);$('play').onclick=()=>{if(!UI.metadata||UI.loading)return;UI.playing=!UI.playing;$('play').textContent=UI.playing?'Ⅱ':'▶';$('play').setAttribute('aria-label',UI.playing?'Pause':'Play')};function animate(now){let elapsed=Math.min((now-UI.lastTick)/1000,.25);UI.lastTick=now;if(UI.playing&&!UI.loading&&!document.hidden){UI.time=Math.min(UI.metadata.duration_s,UI.time+elapsed*Number($('speed').value));if(UI.time>=UI.metadata.duration_s)stop();$('seek').value=UI.time;if(document.activeElement!==$('time'))$('time').value=UI.time.toFixed(2);refresh()}requestAnimationFrame(animate)}requestAnimationFrame(animate);document.addEventListener('visibilitychange',()=>{UI.lastTick=performance.now();if(document.hidden)stop()});window.addEventListener('resize',()=>{for(let p of UI.panels){if(p.type==='scene')drawScene(p);else if(SIGNALS[p.type])drawSignal(p)}});
(async()=>{try{UI.catalog=(await api('/api/catalog')).sort(compareSessions);sessionList();let requested=new URLSearchParams(location.search).get('session');let initial=UI.catalog.find(r=>r.key===requested)||UI.catalog[0];if(initial)await selectSession(initial.key);else notice('No sessions in this collection')}catch(e){notice(e.message)}})();
