import requests


BASE_URL = "http://localhost:8000/processPost"

def test_chatbot1():
    data = {
        "idPost": "10",  
        "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySUQiOiIxIiwic2Vzc2lvbklEIjoiOTg4NzExZTYtNzJlZi00MGYwLTk4NmQtODA0OWVkMDJjZWMwIiwiaWF0IjoxNzMzNjQ4OTgyLCJleHAiOjE3MzQyNTM3ODJ9.6Aw3zklu4G_iaX8ID3TZQwzKSuMfQtjAojmmAzZFLvk"  # Token giả lập
    }

    response = requests.post(f"{BASE_URL}", json=data)

    assert response.status_code == 200, f"Expected status code 200 but got {response.status_code}"
    
    response_json = response.json()
    assert "answer" in response_json, "'answer' key not found in response"
    assert isinstance(response_json["answer"], str), "'answer' should be a string"
    
    print(f"Trả lời từ chatbot: {response_json['answer']}")
