/* ═══════════════════════════════════════════════
   dashboard-ui.js  –  UI helpers & pagination
═══════════════════════════════════════════════ */

'use strict';

/* ── TOAST NOTIFICATION ── */
function showToast(msg, type) {
    type = type || 'info';
    const colors = {
        success: { bg: '#052e16', border: '#16a34a', color: '#4ade80', icon: 'bi-check-circle-fill' },
        danger:  { bg: '#450a0a', border: '#dc2626', color: '#f87171', icon: 'bi-x-circle-fill' },
        info:    { bg: '#0c1a35', border: '#3b82f6', color: '#93c5fd', icon: 'bi-info-circle-fill' },
        warning: { bg: '#1c1205', border: '#d97706', color: '#fcd34d', icon: 'bi-exclamation-triangle-fill' },
    };
    const c = colors[type] || colors.info;

    const t = document.createElement('div');
    t.style.cssText = `
        position:fixed;right:20px;bottom:24px;
        background:${c.bg};border:1px solid ${c.border};color:${c.color};
        padding:12px 18px;border-radius:12px;
        box-shadow:0 8px 32px rgba(0,0,0,.5);
        z-index:9999;display:flex;align-items:center;gap:10px;
        font-size:.85rem;font-weight:500;max-width:320px;
        transition:opacity .3s, transform .3s;
        animation:slideInToast .25s ease;
    `;
    t.innerHTML = `<i class="bi ${c.icon}" style="font-size:1rem"></i><span>${msg}</span>`;
    document.body.appendChild(t);
    setTimeout(() => { t.style.opacity = '0'; t.style.transform = 'translateY(6px)'; }, 3000);
    setTimeout(() => t.remove(), 3300);
}

/* ── REFRESH CAMERA FEED (force img reload) ── */
function refreshCameraFeed(cameraNo) {
    const imgs = document.querySelectorAll('.camera-stream');
    imgs.forEach(img => {
        if (img.src && img.src.includes(`cam${cameraNo}`)) {
            const base = img.src.split('?')[0];
            img.src = base + '?t=' + Date.now();
        }
    });
}

/* ── PAGINATION FACTORY ── */
function setupPagination(prefix, rowClass, pageSize) {
    pageSize = pageSize || 5;
    const prevBtn  = document.getElementById(prefix + '-prev');
    const nextBtn  = document.getElementById(prefix + '-next');
    const pageInfo = document.getElementById(prefix + '-page-info');
    let page = 0;

    function render() {
        const rows  = Array.from(document.getElementsByClassName(rowClass));
        const total = rows.length;
        const pages = total === 0 ? 1 : Math.ceil(total / pageSize);
        if (page >= pages) page = pages - 1;

        rows.forEach((r, i) => {
            r.style.display = (i >= page * pageSize && i < (page + 1) * pageSize) ? '' : 'none';
        });
        if (pageInfo) pageInfo.textContent = total === 0 ? 'No records' : `Page ${page + 1} of ${pages}`;
        if (prevBtn)  prevBtn.disabled  = page === 0 || total === 0;
        if (nextBtn)  nextBtn.disabled  = (page + 1) * pageSize >= total || total === 0;
    }

    if (prevBtn) prevBtn.addEventListener('click', function () { if (page > 0) { page--; render(); } });
    if (nextBtn) nextBtn.addEventListener('click', function () {
        const rows = document.getElementsByClassName(rowClass);
        const total = rows.length;
        if ((page + 1) * pageSize < total) { page++; render(); }
    });

    render();
    return { render };
}

/* ── INIT PAGINATION ON DOM READY ── */
var visitorsPagination = null;
var logsPagination     = null;
var alertsPagination   = null;

document.addEventListener('DOMContentLoaded', function () {
    visitorsPagination = setupPagination('visitor', 'visitor-row', 5);
    logsPagination     = setupPagination('logs',    'log-row',     8);
    alertsPagination   = setupPagination('alerts',  'alert-row',   8);
});

/* ── REC BADGE PULSE ── */
document.addEventListener('DOMContentLoaded', function () {
    const recBadges = document.querySelectorAll('.rec-badge');
    recBadges.forEach(badge => {
        setInterval(() => { badge.style.opacity = badge.style.opacity === '0' ? '1' : '0'; }, 800);
    });
});

/* ── SIDEBAR TOGGLE (mobile) ── */
document.addEventListener('DOMContentLoaded', function () {
    const hamburger = document.getElementById('hamburger');
    const sidebar   = document.getElementById('sidebar');
    if (hamburger && sidebar) {
        hamburger.addEventListener('click', () => sidebar.classList.toggle('sidebar-open'));
    }
});

/* ── KEYFRAME for toast ── */
const styleSheet = document.createElement('style');
styleSheet.textContent = `
@keyframes slideInToast {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0);    }
}`;
document.head.appendChild(styleSheet);

/* ── NOTIFICATIONS & MESSAGES BADGE LOGIC ── */
document.addEventListener('DOMContentLoaded', function () {
    const msgBadge = document.getElementById('msg-badge');
    const msgDropdownBtn = document.getElementById('msgDropdownBtn');
    
    if (localStorage.getItem('messages_read') === 'true') {
        if (msgBadge) msgBadge.style.display = 'none';
    }

    if (msgDropdownBtn) {
        msgDropdownBtn.addEventListener('click', function() {
            localStorage.setItem('messages_read', 'true');
            if (msgBadge) msgBadge.style.display = 'none';
        });
    }

    const notifDropdownBtn = document.getElementById('notifDropdownBtn');
    if (notifDropdownBtn) {
        notifDropdownBtn.addEventListener('click', function() {
            const notifBadge = document.getElementById('notif-badge');
            const sidebarBadge = document.getElementById('sidebar-alert-count');
            
            // Hide immediately for snappy UI
            if (notifBadge) {
                notifBadge.style.display = 'none';
                notifBadge.textContent = '0';
            }
            if (sidebarBadge) {
                sidebarBadge.style.display = 'none';
                sidebarBadge.textContent = '0';
            }

            // Sync with backend session
            const csrfTokenElement = document.querySelector('[name=csrfmiddlewaretoken]');
            const csrfToken = csrfTokenElement ? csrfTokenElement.value : '';
            
            fetch('/mark_alerts_read/', {
                method: 'POST',
                headers: {
                    'X-CSRFToken': csrfToken
                }
            }).catch(e => console.error(e));
        });
    }
});

// --- Camera Settings Modal Functions ---

function openCameraSettings(camId) {
    fetch(`/api/camera/${camId}/`)
        .then(response => {
            if (!response.ok) throw new Error('Camera not found');
            return response.json();
        })
        .then(data => {
            document.getElementById('editCamId').value = data.id;
            document.getElementById('editCamName').value = data.name;
            document.getElementById('editCamUrl').value = data.source_url;
            document.getElementById('editCamType').value = data.camera_type;
            document.getElementById('editCamQuality').value = data.stream_quality;
            document.getElementById('editCamLocation').value = data.location;
            document.getElementById('editCamEnabled').checked = data.is_enabled;
            
            const modal = new bootstrap.Modal(document.getElementById('cameraSettingsModal'));
            modal.show();
        })
        .catch(err => {
            if (typeof showToast === 'function') showToast('Failed to load camera settings');
            else alert('Failed to load camera settings');
        });
}

function saveCameraSettings() {
    const camId = document.getElementById('editCamId').value;
    
    const payload = {
        name: document.getElementById('editCamName').value,
        source_url: document.getElementById('editCamUrl').value,
        camera_type: document.getElementById('editCamType').value,
        stream_quality: document.getElementById('editCamQuality').value,
        location: document.getElementById('editCamLocation').value,
        is_enabled: document.getElementById('editCamEnabled').checked
    };
    
    fetch(`/api/camera/${camId}/`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload)
    })
    .then(r => r.json())
    .then(data => {
        if (data.status === 'success') {
            const modalEl = document.getElementById('cameraSettingsModal');
            const modal = bootstrap.Modal.getInstance(modalEl);
            modal.hide();
            
            if (typeof showToast === 'function') {
                showToast("Settings saved successfully. Refreshing...");
                setTimeout(() => window.location.reload(), 1500);
            } else {
                alert("Settings saved successfully.");
                window.location.reload();
            }
        } else {
            if (typeof showToast === 'function') showToast('Failed to save settings');
            else alert('Failed to save settings');
        }
    })
    .catch(err => {
        if (typeof showToast === 'function') showToast('Error saving settings');
    });
}

function deleteCameraFromSettings() {
    // Hide the settings modal first
    const settingsModalEl = document.getElementById('cameraSettingsModal');
    const settingsModal = bootstrap.Modal.getInstance(settingsModalEl);
    if(settingsModal) settingsModal.hide();
    
    // Show the delete confirmation modal
    const deleteModal = new bootstrap.Modal(document.getElementById('cameraDeleteConfirmModal'));
    deleteModal.show();
}

function confirmDeleteCameraAction() {
    const camId = document.getElementById('editCamId').value;
    
    fetch(`/delete_camera/${camId}/`, {
        method: 'POST',
    }).then(() => {
        if (typeof showToast === 'function') {
            showToast("Camera deleted successfully.");
            setTimeout(() => { window.location.href = '/cameras/'; }, 1000);
        } else {
            window.location.href = '/cameras/';
        }
    }).catch(() => {
        if (typeof showToast === 'function') showToast("Error deleting camera.");
        window.location.href = '/cameras/';
    });
}

