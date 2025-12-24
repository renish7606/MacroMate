import requests

API_KEY = "AJJvZbv1oz1CLIIUEsLpeA==Xs8HlCl5zle7QSJF"

url = "https://api.calorieninjas.com/v1/nutrition"
headers = {
    "X-Api-Key": API_KEY
}
params = {
    "query": "apple"
}

r = requests.get(url, headers=headers, params=params)
print("Status:", r.status_code)
print("Response:", r.text)
