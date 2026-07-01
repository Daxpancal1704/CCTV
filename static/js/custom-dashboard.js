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



});
