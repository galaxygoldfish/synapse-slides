document.addEventListener('DOMContentLoaded', () => {
    const socket = io();

    const startButton = document.getElementById('startButton');
    const stopButton = document.getElementById('stopButton');
    const statusMessagesDiv = document.getElementById('statusMessages');
    const blinkEventsDiv = document.getElementById('blinkEvents');
    const plot1Image = document.getElementById('plot1Image');
    const plot2Image = document.getElementById('plot2Image');
    const plot1Placeholder = document.getElementById('plot1Placeholder');
    const plot2Placeholder = document.getElementById('plot2Placeholder');

    const MAX_MESSAGES = 15; // Max messages to keep in log divs

    function addLogMessage(container, text, type = '') {
        // Remove placeholder if it's the first message
        const placeholder = container.querySelector('p.text-muted');
        if (placeholder) {
            placeholder.remove();
        }

        const p = document.createElement('p');
        p.textContent = text;
        if (type) {
            p.classList.add(`blink-event-${type}`); // e.g., blink-event-single
        }
        
        if (container.firstChild) {
            container.insertBefore(p, container.firstChild);
        } else {
            container.appendChild(p);
        }

        // Trigger animation
        requestAnimationFrame(() => {
            p.classList.add('visible');
        });

        while (container.children.length > MAX_MESSAGES) {
            container.removeChild(container.lastChild);
        }
    }

    startButton.addEventListener('click', () => {
        socket.emit('start_detection');
        addLogMessage(statusMessagesDiv, 'COMMENCING NEURAL LINK...');
        startButton.disabled = true;
        stopButton.disabled = false;
    });

    stopButton.addEventListener('click', () => {
        socket.emit('stop_detection');
        addLogMessage(statusMessagesDiv, 'INITIATING DETECTION SHUTDOWN...');
    });

    socket.on('connect', () => {
        addLogMessage(statusMessagesDiv, 'SYSTEM ONLINE: Connection to server established.');
        console.log('Connected to Socket.IO server');
    });

    socket.on('disconnect', () => {
        addLogMessage(statusMessagesDiv, 'CONNECTION LOST: Disconnected from server.');
        console.log('Disconnected from Socket.IO server');
        startButton.disabled = false;
        stopButton.disabled = true;
    });

    socket.on('status_update', (msg) => {
        addLogMessage(statusMessagesDiv, `[STATUS] ${msg.data}`);
        console.log('Status:', msg.data);
    });

    socket.on('blink_event', (msg) => {
        const message = `[CMD DETECTED] ${msg.type.toUpperCase()} BLINK :: ${msg.action}`;
        addLogMessage(blinkEventsDiv, message, msg.type);
        console.log('Blink Event:', msg);
    });
    
    socket.on('detection_started', () => {
        addLogMessage(statusMessagesDiv, 'NEURAL DETECTION ACTIVE.');
        startButton.disabled = true;
        stopButton.disabled = false;
    });

    socket.on('detection_stopped', () => {
        addLogMessage(statusMessagesDiv, 'NEURAL DETECTION TERMINATED.');
        startButton.disabled = false;
        stopButton.disabled = true;
    });

    socket.on('plot_ready', (msg) => {
        addLogMessage(statusMessagesDiv, `VISUALIZATION GENERATED: ${msg.filename}`);
        const plotUrl = `/static/plots/${msg.filename}?t=${new Date().getTime()}`;
        if (msg.type === 'all_data') {
            plot1Image.src = plotUrl;
            plot1Image.style.display = 'block';
            plot1Placeholder.style.display = 'none';
        } else if (msg.type === 'blink_selected') {
            plot2Image.src = plotUrl;
            plot2Image.style.display = 'block';
            plot2Placeholder.style.display = 'none';
        }
    });
    
    socket.on('clear_plots', () => {
        plot1Image.src = "#";
        plot1Image.style.display = 'none';
        plot1Placeholder.style.display = 'block';
        plot2Image.src = "#";
        plot2Image.style.display = 'none';
        plot2Placeholder.style.display = 'block';
        addLogMessage(statusMessagesDiv, "Previous session visualizations cleared.");
    });
});