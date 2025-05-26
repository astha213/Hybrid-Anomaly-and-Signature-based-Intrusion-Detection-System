import requests

def send_requests(url, num_requests):
    for _ in range(num_requests):
        try:
            response = requests.get(url)
            print(f"Response code: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"Error: {e}")

# Example usage:
url = "http://192.168.1.7:8080/"
num_requests = 10  # Specify the number of requests you want to send
send_requests(url, num_requests)