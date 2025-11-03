import pandas as pd
from pykeen.datasets import CoDExSmall, CoDExMedium
from pykeen.pipeline import pipeline
import torch
from pykeen.training.callbacks import EvaluationTrainingCallback
from pykeen.training import SLCWATrainingLoop
from torch.optim import Adam
from pykeen.evaluation import LCWAEvaluationLoop
from pykeen.models import TransE, DistMult, ComplEx

device_mps = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
device_cpu = torch.device("cpu")

all_results = []
num_epochs = 500
batch_size = 1024


dataset = CoDExSmall()

training = dataset.training
validation = dataset.validation
testing = dataset.testing

models = {
    "TransE": TransE(triples_factory=training, embedding_dim=100),
    "DistMult": DistMult(triples_factory=training, embedding_dim=100),
    "ComplEx": ComplEx(triples_factory=training, embedding_dim=100),
}


for model_name,model in models.items():
    
    device = device_mps if model_name != "ComplEx" else device_cpu
    model.to(device)

    optimizer = Adam(params=model.get_grad_params())

    training_loop_local = SLCWATrainingLoop(
        model=model,
        triples_factory=training,
        optimizer=optimizer,
    )
    
    eval_callback = EvaluationTrainingCallback(
        evaluation_triples=validation.mapped_triples,
        prefix="validation",
        additional_filter_triples=[
            training.mapped_triples,
            validation.mapped_triples,
        ],
    )



    losses_per_epoch = training_loop_local.train(
        triples_factory=training,
        num_epochs=num_epochs,
        batch_size=batch_size,
        callbacks=[eval_callback],
    )
    
    evaluation_loop = LCWAEvaluationLoop(
        model=model,
        triples_factory=testing,
        additional_filter_triples=[
            training,
            validation,
        ],
    )

    results = evaluation_loop.evaluate()
    
    mrr = results.get_metric('mean_reciprocal_rank')
    hits_at_1 = results.get_metric('hits@1')
    hits_at_5 = results.get_metric('hits@5')
    hits_at_10 = results.get_metric('hits@10')
    
    all_results.append({
        "Model": model_name,
        "MRR": mrr,
        "Hits@1": hits_at_1,
        "Hits@5": hits_at_5,
        "Hits@10": hits_at_10
    })

df = pd.DataFrame(all_results)
df.to_csv("codexsmall_results.csv", index=False)
print("Results saved to codexsmall_results.csv")
