chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.type === 'SYNC_POSITIONS') {
        chrome.storage.local.get(['apiUrl'], (result) => {
            // Use a new endpoint for syncing: /api/sync
            // But we can reuse the base URL logic.
            let baseUrl = result.apiUrl || 'http://localhost:3000/api/trades';
            // Convert /api/trades to /api/sync
            // Assumption: default is .../api/trades, we want .../api/sync
            const syncUrl = baseUrl.replace('/trades', '/sync');

            // console.log('Syncing positions to:', syncUrl);

            fetch(syncUrl, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(message.data)
            })
                .then(response => {
                    if (!response.ok) {
                        // Silent fail on sync errors to avoid spamming logs, or log only once in a while
                        // console.warn('Sync failed:', response.status);
                    }
                })
                .catch(error => {
                    // console.error('Error syncing positions:', error);
                });
        });
    } else if (message.type === 'NEW_TRADE') {
        // ... legacy code for notifications (can keep or remove) ...
        chrome.storage.local.get(['apiUrl'], (result) => {
            const apiUrl = result.apiUrl || 'http://localhost:3000/api/trades';
            // console.log('Sending trade to:', apiUrl);
            // ... existing fetch logic ...
             fetch(apiUrl, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(message.data)
            })
                .then(response => {
                    // console.log('Fetch response status:', response.status);
                    if (!response.ok) {
                        return response.text().then(text => {
                            throw new Error(`HTTP ${response.status}: ${text}`);
                        });
                    }
                    return response.json();
                })
                .then(data => {
                    console.log('Trade sent to AlphaOS SUCCESSFULLY');
                })
                .catch(error => {
                    console.error('Error sending trade to AlphaOS:', error);
                });
        });
    }
    return true; 
});
