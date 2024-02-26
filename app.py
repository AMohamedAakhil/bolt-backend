from flask import Flask, request, jsonify
from diffusers import masker, utils, ip_adapter

app = Flask(__name__)

messages = []

# Endpoint to receive masked chat messages
@app.route('/receive_masked_message', methods=['GET'])
def receive_masked_message():
    masked_messages = masker.mask(messages)  # Masking sensitive information in messages
    return jsonify(masked_messages)

@app.route('/send_message_with_ip', methods=['POST'])
def send_message_with_ip():
    data = request.get_json()
    message = data.get('message')
    if message:
        client_ip = ip_adapter.get_client_ip(request)  # Retrieve client's IP address
        message_with_ip = utils.add_ip_to_message(message, client_ip)  # Append IP address to message
        messages.append(message_with_ip)
        return jsonify({'status': 'Message sent successfully with IP address'})
    else:
        return jsonify({'status': 'Error: No message provided'})

@app.route('/manipulate_message', methods=['POST'])
def manipulate_message():
    data = request.get_json()
    message = data.get('message')
    if message:
        # Example of complex message manipulation using utilities
        manipulated_message = utils.perform_complex_manipulation(message)
        messages.append(manipulated_message)
        return jsonify({'status': 'Message manipulated and added successfully'})
    else:
        return jsonify({'status': 'Error: No message provided'})

@app.route('/convert_to_uppercase', methods=['POST'])
def convert_to_uppercase():
    data = request.get_json()
    message = data.get('message')
    if message:
        uppercase_message = message.upper()
        messages.append(uppercase_message)
        return jsonify({'status': 'Message converted to uppercase and added successfully'})
    else:
        return jsonify({'status': 'Error: No message provided'})

@app.route('/reverse_message', methods=['POST'])
def reverse_message():
    data = request.get_json()
    message = data.get('message')
    if message:
        reversed_message = message[::-1]
        messages.append(reversed_message)
        return jsonify({'status': 'Message reversed and added successfully'})
    else:
        return jsonify({'status': 'Error: No message provided'})

if __name__ == '__main__':
    app.run(debug=True)
