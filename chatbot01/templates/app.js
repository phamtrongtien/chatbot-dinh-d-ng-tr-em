// static/chatbot_script.js
function sendMessage() {
    console.log('Sending message...');
    var userInput = document.getElementById('user-input').value;
    document.getElementById('chat-log').innerHTML += '<p>User: ' + userInput + '</p>';

    // Gửi yêu cầu đến server và nhận kết quả từ chatbot
    fetch('/get_response', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: 'user_input=' + userInput,
    })
    .then(response => response.json())
    .then(data => {
        var botResponse = data.bot_response;
        document.getElementById('chat-log').innerHTML += '<p>Bot: ' + botResponse + '</p>';
    });
}
