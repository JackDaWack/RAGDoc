document.getElementById('submitQuery').addEventListener('click', query);
document.getElementById('responseContent').addEventListener('click', response);

async function query() {
    try {
        const queryInput = document.getElementById('queryInput').value;
        const response = await fetch('/query', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ question: queryInput })
        });
        const data = await response.json();
    } catch (error) {
        console.error('Error:', error);
    }
}
async function response() {
    try {
        const response = await fetch('/response', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });
        const data = await response.json();
        document.getElementById('responseContent').textContent = data.response;
    } catch (error) {
        console.error('Error:', error);
    }
}