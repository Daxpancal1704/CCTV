/* ═══════════════════════════════════════════════
   dashboard.js  –  Core live-data polling
═══════════════════════════════════════════════ */

'use strict';

/* ── CLOCK ── */
function updateClock() {
    const el = document.getElementById('clock');
    if (el) el.textContent = new Date().toLocaleString();
}
setInterval(updateClock, 1000);
updateClock();

/* ── FULLSCREEN ── */
function toggleFullscreen() {
    if (!document.fullscreenElement) {
        document.documentElement.requestFullscreen();
    } else {
        document.exitFullscreen();
    }
}

/* ── THEME TOGGLE ── */
(function () {
    const saved = localStorage.getItem('cctv-theme') || 'dark';
    document.documentElement.setAttribute('data-theme', saved);
    document.documentElement.setAttribute('data-bs-theme', saved);
    const toggle = document.getElementById('themeToggle');
    if (toggle) toggle.checked = (saved === 'light');
})();

const themeToggle = document.getElementById('themeToggle');
if (themeToggle) {
    themeToggle.addEventListener('change', function () {
        const theme = this.checked ? 'light' : 'dark';
        document.documentElement.setAttribute('data-theme', theme);
        document.documentElement.setAttribute('data-bs-theme', theme);
        localStorage.setItem('cctv-theme', theme);
    });
}

/* ── FACE COUNT ── */
function updateFaceCount() {
    fetch('/face_count/')
        .then(r => r.json())
        .then(d => {
            const el = document.getElementById('face-count');
            if (el) el.textContent = d.count || 0;
        })
        .catch(() => {});
}
setInterval(updateFaceCount, 2000);
updateFaceCount();

/* ── PEOPLE COUNT ── */
function updatePeopleCount() {
    fetch('/people_count/')
        .then(r => r.json())
        .then(d => {
            const el = document.getElementById('people-count');
            if (el) el.textContent = d.count || 0;
        })
        .catch(() => {});
}
setInterval(updatePeopleCount, 2000);
updatePeopleCount();

/* ── OCCUPANCY / ENTRY / EXIT ── */
function updateOccupancy() {
    fetch('/occupancy/')
        .then(r => r.json())
        .then(d => {
            const entry   = document.getElementById('entry-count');
            const exit    = document.getElementById('exit-count');
            const occ     = document.getElementById('occupancy-count');
            const events  = document.getElementById('event-count');
            const visitors = document.getElementById('visitor-count');
            if (entry)  entry.textContent  = d.entry   || 0;
            if (exit)   exit.textContent   = d.exit    || 0;
            if (occ)    occ.textContent    = d.occupancy || 0;
            if (events) events.textContent = (d.entry || 0) + (d.exit || 0);
            if (visitors) visitors.textContent = d.visitors_today || 0;
        })
        .catch(() => {});
}
setInterval(updateOccupancy, 3000);
updateOccupancy();

/* ── CAMERA STATUS ── */
// MOVED TO INLINE SCRIPT IN INDEX.HTML TO PREVENT CACHE ISSUES
function updateCameraStatus() {
    // Disabled here. Runs from index.html now.
}
// setInterval(updateCameraStatus, 5000);
// updateCameraStatus();


/* ── ALERT HISTORY ── */
function updateAlerts() {
    fetch('/alerts/')
        .then(r => r.json())
        .then(d => {
            const body = document.getElementById('alert-history-body');
            if (!body) return;
            body.innerHTML = '';
            (d.alerts || []).forEach(alert => {
                const typeClass = {
                    'Motion':          'alert-motion',
                    'Face':            'alert-face',
                    'Unknown Face':    'alert-face',
                    'Intrusion':       'alert-intrusion',
                    'Blacklisted Person': 'alert-intrusion',
                    'Camera Offline':  'alert-motion',
                }[alert.alert_type] || 'alert-default';

                const icon = {
                    'Motion':          'bi-arrows-move',
                    'Face':            'bi-person-fill-slash',
                    'Unknown Face':    'bi-person-fill-slash',
                    'Intrusion':       'bi-exclamation-triangle-fill',
                    'Blacklisted Person': 'bi-ban',
                    'Camera Offline':  'bi-camera-video-off-fill',
                }[alert.alert_type] || 'bi-bell';

                const tr = document.createElement('tr');
                tr.className = 'alert-row';
                tr.innerHTML = `
                    <td><span class="alert-type-badge ${typeClass}">
                        <i class="bi ${icon} me-1"></i>${alert.alert_type}
                    </span></td>
                    <td style="font-size:.8rem">${alert.message}</td>
                    <td style="font-family:'JetBrains Mono',monospace;font-size:.75rem">${alert.created_at}</td>
                `;
                body.appendChild(tr);
            });

            // Update sidebar & topbar badge to show unread count (synced via session)
            const count = d.unread_count !== undefined ? d.unread_count : (d.alerts || []).length;
            const sidebarBadge = document.getElementById('sidebar-alert-count');
            const notifBadge   = document.getElementById('notif-badge');
            
            if (sidebarBadge) {
                sidebarBadge.textContent = count > 0 ? count : '0';
                sidebarBadge.style.display = count > 0 ? 'inline-block' : 'none';
            }
            if (notifBadge) {
                notifBadge.textContent = count > 0 ? count : '0';
                notifBadge.setAttribute('data-total', count);
                notifBadge.style.display = count > 0 ? 'inline-block' : 'none';
            }

            // Re-render pagination if it exists
            if (typeof alertsPagination !== 'undefined' && alertsPagination) alertsPagination.render();
        })
        .catch(() => {});
}
setInterval(updateAlerts, 5000);
updateAlerts();

/* ── SNAPSHOT (called from camera buttons) ── */
function captureSnapshot(cameraNo) {
    fetch(`/snapshot/?camera=${cameraNo}`)
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                if (typeof showToast === 'function') {
                    showToast(`Snapshot successfully captured from Camera ${cameraNo}!`, 'success');
                }
            } else {
                if (typeof showToast === 'function') {
                    showToast(`Failed to capture snapshot: ${data.message}`, 'error');
                }
            }
        })
        .catch(err => {
            console.error('Snapshot error:', err);
            if (typeof showToast === 'function') {
                showToast(`Error connecting to camera.`, 'error');
            }
        });
}

/* ── SNAPSHOT DOWNLOAD MODAL ── */
function openDownloadModal(snapshotId) {
    document.getElementById('download-snap-id').value = snapshotId;
    const modal = new bootstrap.Modal(document.getElementById('downloadSnapshotModal'));
    modal.show();
}

function triggerDownload(format) {
    const snapId = document.getElementById('download-snap-id').value;
    if (snapId) {
        window.location.href = `/download_snapshot/${snapId}/?format=${format}`;
        // Hide the modal after initiating download
        const modalEl = document.getElementById('downloadSnapshotModal');
        const modal = bootstrap.Modal.getInstance(modalEl);
        if (modal) modal.hide();
    }
}


/* ── ADD CAMERA FORM ── */
document.addEventListener('DOMContentLoaded', function () {
    const form = document.getElementById('add-camera-form');
    if (form) {
        form.addEventListener('submit', function (e) {
            e.preventDefault();
            const csrfToken = form.querySelector('[name=csrfmiddlewaretoken]').value;
            const payload = {
                name:         document.getElementById('new-camera-name').value,
                source_url:   document.getElementById('new-camera-url').value,
                camera_type:  document.getElementById('new-camera-type').value,
            };
            fetch('/add_camera/', {
                method:  'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken':  csrfToken,
                },
                body: JSON.stringify(payload),
            })
            .then(r => r.json())
            .then(d => {
                if (typeof showToast === 'function') showToast(d.message || 'Camera added', 'success');
                setTimeout(() => location.reload(), 1200);
            })
            .catch(() => {
                if (typeof showToast === 'function') showToast('Failed to add camera', 'danger');
            });
        });
    }
});
