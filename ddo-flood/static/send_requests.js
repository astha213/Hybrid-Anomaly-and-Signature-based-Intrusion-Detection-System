// Global variables
let isAttacking = false;
let attackInterval;
let requestCount = 0;
let lastRequestTime = Date.now();
let requestsPerSecond = 0;
let maxRequests = Math.floor(Math.random() * (800 - 600 + 1)) + 600; // Random between 600-800
let isBlocked = false; // Add flag to track if IP is already blocked

// Function to generate attack data
function generateAttackData() {
    const currentTime = Date.now();
    const timeDiff = (currentTime - lastRequestTime) / 1000; // Convert to seconds
    requestsPerSecond = timeDiff > 0 ? 1 / timeDiff : 0;
    lastRequestTime = currentTime;

    // Calculate intensity based on progress towards maxRequests
    const progress = requestCount / maxRequests;
    const intensity = Math.min(1, progress * 2); // Ramp up intensity over first half of requests

    // Focus on DoS-relevant features with gradual intensity
    return {
        duration: timeDiff,
        protocol_type: 1,  // HTTP
        service: 1,        // web
        flag: 1,          // normal
        src_bytes: Math.floor(1000 * intensity),  // Gradually increase packet size
        dst_bytes: Math.floor(500 * intensity),   // Gradually increase response size
        hot: intensity > 0.5 ? 1 : 0,           // Start accessing system directories after 50% progress
        logged_in: 0,
        num_compromised: 0,
        count: requestCount,        // Number of connections to the same host
        srv_count: requestCount,    // Number of connections to the same service
        serror_rate: 0,
        srv_serror_rate: 0,
        rerror_rate: 0,
        srv_rerror_rate: 0,
        same_srv_rate: 1,          // High rate of same service
        diff_srv_rate: 0,
        srv_diff_host_rate: 0,
        dst_host_count: requestCount,
        dst_host_srv_count: requestCount,
        dst_host_same_srv_rate: 1,
        dst_host_diff_srv_rate: 0,
        dst_host_same_src_port_rate: 1,
        dst_host_srv_diff_host_rate: 0,
        dst_host_serror_rate: 0,
        dst_host_srv_serror_rate: 0,
        dst_host_rerror_rate: 0,
        dst_host_srv_rerror_rate: 0
    };
}

// Function to show custom alert
function showCustomAlert() {
    const alertDiv = document.createElement('div');
    alertDiv.style.cssText = `
        position: fixed;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        background: white;
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        z-index: 1000;
        min-width: 300px;
        text-align: left;
    `;

    alertDiv.innerHTML = `
        <div style="font-size: 16px; margin-bottom: 10px;">
            <strong>192.168.1.7:8080 says</strong>
        </div>
        <div style="margin-bottom: 15px;">
            <span style="color: red;">⛔ Your IP (192.168.1.7) has been blocked!</span>
        </div>
        <div style="margin-bottom: 5px;">Reason: DoS Attack Detected</div>
        <div style="margin-bottom: 5px;">Total Requests: ${requestCount}</div>
        <div style="margin-bottom: 15px;">Final Rate: ${requestsPerSecond.toFixed(1)} req/sec</div>
        <div style="margin-bottom: 20px;">This IP will be blocked from making further requests.</div>
        <div style="text-align: right;">
            <button style="
                background: #8B4513;
                color: white;
                border: none;
                padding: 8px 20px;
                border-radius: 20px;
                cursor: pointer;
            ">OK</button>
        </div>
    `;

    // Add overlay
    const overlay = document.createElement('div');
    overlay.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0,0,0,0.5);
        z-index: 999;
    `;

    document.body.appendChild(overlay);
    document.body.appendChild(alertDiv);

    // Handle OK button click
    const okButton = alertDiv.querySelector('button');
    okButton.onclick = function() {
        overlay.remove();
        alertDiv.remove();
        // Reset the page state after alert is closed
        const startButton = document.getElementById('start-btn');
        if (startButton) {
            startButton.disabled = false;
        }
    };
}

// Function to update displays
function updateDisplays() {
    const countDisplay = document.getElementById('request-count');
    const rateDisplay = document.getElementById('rate-display');
    if (countDisplay) {
        countDisplay.textContent = `Total Requests: ${requestCount}`;
    }
    if (rateDisplay) {
        rateDisplay.textContent = `Current Rate: ${requestsPerSecond.toFixed(1)} req/sec`;
    }
}

// Function to stop the attack
function stopAttack() {
    if (isAttacking) {
        isAttacking = false;
        clearInterval(attackInterval);
        const messageDiv = document.getElementById('message');
        const statusDiv = document.createElement('div');
        statusDiv.textContent = `Attack simulation completed. Total requests sent: ${requestCount} | Final Rate: ${requestsPerSecond.toFixed(1)} req/sec`;
        messageDiv.insertBefore(statusDiv, messageDiv.firstChild);
        console.log('Attack simulation completed');
        
        // Show block message if not already shown
        if (!isBlocked) {
            showCustomAlert();
            isBlocked = true;
        }

        // Disable the start button
        const startButton = document.getElementById('start-btn');
        if (startButton) {
            startButton.disabled = true;
        }
    }
}

// Function to send a single request
async function sendRequest() {
    if (isBlocked) return; // Don't send requests if already blocked
    
    try {
        // Check if we've reached the maximum requests
        if (requestCount >= maxRequests) {
            stopAttack();
            return;
        }

        // Generate attack data with high request rate pattern
        const attackData = generateAttackData();
        
        const response = await fetch('/receive', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify([attackData])  // Send as array for model batch processing
        });

        // Check if IP is blocked
        if (response.status === 403) {
            if (!isBlocked) {  // Only show block message once
                isBlocked = true;
                stopAttack();
                clearInterval(attackInterval);
                showCustomAlert();
            }
            return;
        }

        // Update counters and display
        requestCount++;
        updateDisplays();

        const result = await response.json();
        
        if (result.attack_detected && result.attack_detected.length > 0) {
            // Display attack alert on the website
            const messageDiv = document.getElementById('message');
            const alertDiv = document.createElement('div');
            alertDiv.className = 'alert';
            
            const currentTime = new Date().toLocaleTimeString('en-US', {
                hour: 'numeric',
                minute: '2-digit',
                second: '2-digit',
                hour12: true
            });
            
            alertDiv.innerHTML = `
                🚨 DoS Attack Detected at ${currentTime}
                <br>Request Count: ${requestCount}
                <br>Request Rate: ${requestsPerSecond.toFixed(1)} req/sec
                <br>Connection Duration: ${attackData.duration.toFixed(3)}s
                <br>Status: High-frequency request pattern detected
            `;
            
            messageDiv.insertBefore(alertDiv, messageDiv.firstChild);
            
            // Keep only the last 5 alerts
            while (messageDiv.children.length > 5) {
                messageDiv.removeChild(messageDiv.lastChild);
            }
        }
    } catch (error) {
        console.error('Error:', error);
    }
}

// Function to start the attack
function startAttack() {
    if (!isAttacking && !isBlocked) { // Check if IP is not blocked
        isAttacking = true;
        requestCount = 0;
        lastRequestTime = Date.now();
        requestsPerSecond = 0;
        maxRequests = Math.floor(Math.random() * (800 - 600 + 1)) + 600; // Random between 600-800
        
        const messageDiv = document.getElementById('message');
        messageDiv.innerHTML = '';  // Clear previous messages
        
        // Create or update request count display
        const countDisplay = document.getElementById('request-count');
        if (!countDisplay) {
            const countDiv = document.createElement('div');
            countDiv.id = 'request-count';
            countDiv.style.fontSize = '18px';
            countDiv.style.fontWeight = 'bold';
            countDiv.style.marginBottom = '10px';
            document.querySelector('.container').insertBefore(countDiv, messageDiv);
        }
        updateDisplays();
        
        // Send requests at a fixed interval to ensure we reach the target
        const interval = 10; // 10ms between requests = 100 requests per second
        attackInterval = setInterval(sendRequest, interval);
        
        console.log('DoS Attack simulation started with target of', maxRequests, 'requests');
    }
}

// Wait for the page to load
document.addEventListener('DOMContentLoaded', function() {
    console.log('Page loaded, setting up listeners');
    
    // Get the start button
    const startButton = document.getElementById('start-btn');
    
    if (!startButton) {
        console.error('Could not find start button!');
        return;
    }

    // Add click handler
    startButton.onclick = function() {
        console.log('Start button clicked');
        startButton.disabled = true;
        startAttack();
    };
});
