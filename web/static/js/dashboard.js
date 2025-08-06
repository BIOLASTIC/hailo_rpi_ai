document.addEventListener('DOMContentLoaded', function () {
    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${wsProtocol}//${window.location.host}/ws`;
    const socket = new WebSocket(wsUrl);

    const detectionsList = document.getElementById('detections-list');

    socket.onopen = () => console.log('[WebSocket] Connection established.');
    socket.onclose = () => console.log('[WebSocket] Connection closed.');
    socket.onerror = (error) => console.error(`[WebSocket] Error: ${error.message}`);

    socket.onmessage = (event) => {
        const message = JSON.parse(event.data);
        if (message.type === 'ai_detections') {
            updateDetectionsList(message.data);
        }
    };

    function updateDetectionsList(detections) {
        // Clear current list
        detectionsList.innerHTML = '';

        if (detections.length === 0) {
            detectionsList.innerHTML = '<li class="list-group-item text-muted">No objects detected.</li>';
            return;
        }

        detections.forEach(det => {
            const listItem = document.createElement('li');
            listItem.className = 'list-group-item d-flex justify-content-between align-items-center';
            
            const label = document.createElement('span');
            label.className = 'fw-bold';
            label.textContent = det.label;

            const confidence = document.createElement('span');
            confidence.className = 'badge bg-primary rounded-pill';
            confidence.textContent = `${(det.confidence * 100).toFixed(1)}%`;
            
            listItem.appendChild(label);
            listItem.appendChild(confidence);
            detectionsList.appendChild(listItem);
        });
    }
});