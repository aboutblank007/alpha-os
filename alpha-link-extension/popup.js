document.addEventListener('DOMContentLoaded', () => {
    const apiUrlInput = document.getElementById('apiUrl');
    const saveButton = document.getElementById('save');
    const testButton = document.getElementById('test');
    const statusDiv = document.getElementById('status');

    // Load saved URL
    chrome.storage.local.get(['apiUrl'], (result) => {
        if (result.apiUrl) {
            apiUrlInput.value = result.apiUrl;
        } else {
            apiUrlInput.value = 'http://localhost:3000/api/trades';
        }
    });

    // Save URL
    saveButton.addEventListener('click', () => {
        const apiUrl = apiUrlInput.value;
        chrome.storage.local.set({ apiUrl }, () => {
            showStatus('Saved!', 'success');
        });
    });

    // Test Connection
    testButton.addEventListener('click', () => {
        const apiUrl = apiUrlInput.value;
        showStatus('Testing...', '');

        // Send a dummy trade to test the endpoint
        const dummyTrade = {
            symbol: 'TEST',
            side: 'buy',
            entry_price: 100,
            quantity: 1,
            status: 'closed',
            pnl_net: 0,
            notes: 'Connection Test'
        };

        fetch(apiUrl, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(dummyTrade)
        })
            .then(response => {
                if (response.ok) {
                    return response.json();
                }
                throw new Error('Network response was not ok.');
            })
            .then(data => {
                showStatus('Success! Check Dashboard.', 'success');
            })
            .catch(error => {
                showStatus('Error: ' + error.message, 'error');
            });
    });

    function showStatus(msg, type) {
        statusDiv.textContent = msg;
        statusDiv.className = type;
        if (type === 'success') {
            setTimeout(() => {
                statusDiv.textContent = '';
                statusDiv.className = '';
            }, 3000);
        }
    }
});
