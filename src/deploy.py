from azureml.core import Workspace, Model
from azureml.core.environment import Environment
from azureml.core.model import InferenceConfig
from azureml.core.webservice import AciWebservice

# Se connecter à l'espace de travail Azure ML
ws = Workspace.from_config("src/config.json")

# Charger le modèle enregistré
model = Model(ws, name="titanic_model")

# Chemin correct vers le fichier conda.yaml
conda_file_path = "models/conda.yaml"  # Assurez-vous que ce chemin est correct

# Créer l'environnement d'exécution à partir du fichier conda.yaml
env = Environment.from_conda_specification(name="mlflow-env", file_path=conda_file_path)

# Configurer l'inférence
inference_config = InferenceConfig(entry_script="src/score.py", environment=env)

# Configurer le déploiement
aci_config = AciWebservice.deploy_configuration(
    cpu_cores=1,  # Nombre de cœurs CPU
    memory_gb=1,  # Quantité de mémoire RAM en Go
    tags={"model": "titanic", "framework": "scikit-learn"},
    description="Service de prédiction de survie Titanic"
)

# Déployer le service web
service_name = "titanic-survival-service"
service = Model.deploy(
    workspace=ws,
    name=service_name,
    models=[model],
    inference_config=inference_config,
    deployment_config=aci_config
)
service.wait_for_deployment(show_output=True)

# Afficher l'état du service
print(service.state)
print("Scoring URI: " + service.scoring_uri)
