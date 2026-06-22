// Custom Dashboard interactivity
function showToast(msg, timeout=3500){
  const t = document.createElement('div');
  t.className='toast';
  t.textContent = msg;
  document.body.appendChild(t);
  setTimeout(()=>{t.style.opacity='0';t.style.transform='translateY(6px)';},timeout-300);
  setTimeout(()=>t.remove(),timeout);
}

// Snapshot handler (uses existing endpoint)
document.addEventListener('DOMContentLoaded', ()=>{
  const btn = document.getElementById('snapshot-btn');
  if(btn){
    btn.addEventListener('click', ()=>{
      fetch('/snapshot/')
        .then(r=>r.json())
        .then(d=>showToast(d.message || 'Snapshot captured'))
        .catch(()=>showToast('Snapshot failed'));
    })
  }

  // Enhance camera status display (handles missing time)
  function updateCameraStatusUI(data){
    const c1 = document.getElementById('camera1-status');
    const c2 = document.getElementById('camera2-status');
    if(c1) { c1.textContent = data.camera1; c1.className = 'status-badge ' + (data.camera1==='Online' ? 'status-online' : 'status-offline'); }
    if(c2) { c2.textContent = data.camera2; c2.className = 'status-badge ' + (data.camera2==='Online' ? 'status-online' : 'status-offline'); }
  }

  // start a lightweight poll for camera status
  setInterval(()=>{
    fetch('/camera_status/').then(r=>r.json()).then(updateCameraStatusUI).catch(()=>{});
  },5000);

});
