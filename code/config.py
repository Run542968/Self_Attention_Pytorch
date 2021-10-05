##
params_set={
    "epochs":5,
	"use_regularization":"True",
	"C":0.03,
	"clip":"True",
	"use_embeddings":"False",
	"attention_hops":10
}

##
model_params={
    "batch_size": 512,
    "vocab_size": 20000,
    "timesteps": 200,
    "lstm_hidden_dimension": 50,
    "d_a": 100
}