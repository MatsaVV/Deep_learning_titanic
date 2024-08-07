import requests
import json

# URL du service
scoring_uri = "http://20.19.151.251/score"

# Exemple de données
data = {
    "data": [
        [1, 22, 1, 0, 7.25, 0, 2, 0, 1]
    ]
}

# Convertir les données en JSON
input_data = json.dumps(data)

# Définir les en-têtes
headers = {"Content-Type": "application/json"}

# Envoyer la requête POST et afficher la réponse
response = requests.post(scoring_uri, data=input_data, headers=headers)
print(response.json())
