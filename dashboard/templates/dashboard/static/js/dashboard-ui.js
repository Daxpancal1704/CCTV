// Dashboard UI interactive helpers
(function(){
    // Animated counter helper
    function animateCounter(el, start, end, duration){
        var range = end - start;
        var minTimer = 50;
        var stepTime = Math.max(Math.floor(duration / range), minTimer);
        var current = start;
        var step = end > start ? 1 : -1;
        var timer = setInterval(function(){
            current += step;
            el.textContent = current;
            if(current == end){
                clearInterval(timer);
            }
        }, stepTime);
    }

    // Initialize counters on page load
    function initCounters(){
        var faceEl = document.getElementById('face-count');
        var peopleEl = document.getElementById('people-count');
        if(faceEl) animateCounter(faceEl, 0, parseInt(faceEl.textContent)||0, 800);
        if(peopleEl) animateCounter(peopleEl, 0, parseInt(peopleEl.textContent)||0, 800);
    }

    // Collapsible card toggles
    function initCollapsibles(){
        document.querySelectorAll('.card .card-header').forEach(function(header){
            if(header.classList.contains('no-toggle')) return;
            var btn = document.createElement('span');
            btn.className = 'float-end section-toggle';
            btn.innerHTML = '<span class="chev">▾</span>';
            header.appendChild(btn);
            btn.addEventListener('click', function(){
                var card = header.closest('.card');
                if(!card) return;
                card.classList.toggle('section-collapsed');
                var body = card.querySelector('.card-body');
                if(body) body.style.display = card.classList.contains('section-collapsed') ? 'none' : '';
            });
        });
    }

    // Floating snapshot quick button (wire to existing snapshot endpoint)
    function initFab(){
        var fab = document.createElement('div');
        fab.className = 'fab';
        fab.title = 'Quick snapshot';
        fab.innerHTML = '📸';
        document.body.appendChild(fab);
        fab.addEventListener('click', function(){
            fetch('/snapshot/').then(r=>r.json()).then(d=>{
                var msg = d.message || 'Snapshot saved';
                // small toast
                var el = document.createElement('div');
                el.className = 'alert alert-success position-fixed';
                el.style.right='24px'; el.style.bottom='90px'; el.style.zIndex=1200;
                el.textContent = msg;
                document.body.appendChild(el);
                setTimeout(()=>el.remove(),2000);
            }).catch(()=>{
                showToast('Snapshot failed');
            });
        });
    }

    // Expose initializer
    document.addEventListener('DOMContentLoaded', function(){
        try{ initCounters(); }catch(e){}
        try{ initCollapsibles(); }catch(e){}
        try{ initFab(); }catch(e){}
    });
})();
