// attack_alerts.js
document.addEventListener('DOMContentLoaded', function() {
    // Connect to port 3000
    const socket = io('http://localhost:3000');
    const alertsContainer = document.getElementById('attack-alerts');
    
    // Create alert element
    function createAlertElement(alertData) {
        const alertDiv = document.createElement('div');
        alertDiv.className = 'alert alert-danger alert-dismissible fade show';
        alertDiv.role = 'alert';
        
        const timestamp = new Date(alertData.timestamp).toLocaleString();
        const details = alertData.details;
        
        alertDiv.innerHTML = `
            <strong>🚨 Attack Detected!</strong>
            <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
            <hr>
            <p><strong>Time:</strong> ${timestamp}</p>
            <p><strong>Source Bytes:</strong> ${details.src_bytes}</p>
            <p><strong>Destination Bytes:</strong> ${details.dst_bytes}</p>
            <p><strong>Request Count:</strong> ${details.count}</p>
        `;
        
        return alertDiv;
    }
    
    // Handle attack alerts
    socket.on('attack_alert', function(data) {
        const alertElement = createAlertElement(data);
        alertsContainer.prepend(alertElement);
        
        // Auto-dismiss after 30 seconds
        setTimeout(() => {
            alertElement.classList.remove('show');
            setTimeout(() => alertElement.remove(), 150);
        }, 30000);
    });
}); 