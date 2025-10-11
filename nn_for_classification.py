import torch
import copy
from sklearn.base import BaseEstimator, ClassifierMixin

"""Class for Neural Network"""
class NNForClassification(torch.nn.Module):
    def __init__(self, input_size, hidden_layers_sizes=[64, 32, 16], n_classes=2, dropout_prob=0.2, regularization_type=None, lambda_reg=0.1):
        super().__init__()
        self._n_classes = n_classes

        layers = [torch.nn.Linear(in_features=input_size, out_features=hidden_layers_sizes[0])]
        for i in range( len(hidden_layers_sizes) - 1 ):
            layers.append( torch.nn.Dropout(p=dropout_prob) )
            layers.append( torch.nn.ReLU() )
            layers.append( torch.nn.Linear(in_features=hidden_layers_sizes[i], out_features=hidden_layers_sizes[i+1]) )
        layers.append( torch.nn.Dropout(p=dropout_prob) )
        layers.append( torch.nn.ReLU() )
        layers.append ( torch.nn.Linear(in_features=hidden_layers_sizes[-1], out_features=n_classes) )
        self._layer_stack = torch.nn.Sequential( *layers )
        
        # self.output_activation = torch.nn.Softmax(dim=1)
        self._loss_fn = torch.nn.CrossEntropyLoss()
        self._reg_type = regularization_type
        self._lambda_reg = lambda_reg
        self._train_losses = []
        self._valid_losses = []
        self._best_valid_loss = float('inf')
        self._break_epoch = 1

    def forward(self, x):
        x = self.layer_stack(x)
        # x = self.output_activation(x)
        return x
    
    @property
    def layer_stack(self):
        return self._layer_stack

    @property
    def loss_fn(self):
        return self._loss_fn
    
    @property
    def reg_type(self):
        return self._reg_type
    
    @property
    def lambda_reg(self):
        return self._lambda_reg
    
    @property
    def train_losses(self):
        return self._train_losses
    
    @property
    def valid_losses(self):
        return self._valid_losses   
    
    @property
    def best_valid_loss(self):
        return self._best_valid_loss
    
    @best_valid_loss.setter
    def best_valid_loss(self, value):
        self._best_valid_loss = value

    @property   
    def break_epoch(self):
        return self._break_epoch

    @break_epoch.setter
    def break_epoch(self, value):
        self._break_epoch = value


"""Neural Network training process"""
def nn_train(model, optimizer, X_train, y_train, X_valid, y_valid, epochs=1000, patience=100, random_state=68):
    torch.manual_seed(random_state)
    best_valid_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(1, epochs+1):
        model.train()
        
        # Forward pass
        y_train_logits = model(X_train).squeeze() # logits for train data
        # y_train_probs = torch.softmax(y_train_logits, dim=1)
        # y_train_pred = y_train_probs.argmax(dim=1) # probabilities -> labels (predictions)
        
        # Calculate loss (Inside CrossEntropy there is Softmax calculateda so as input we put logits)
        loss = model.loss_fn(y_train_logits, y_train)

        # Apply L1 regularization
        if model.reg_type == 'L1':
            l1_norm = sum(param.abs().sum() for param in model.parameters())
            loss += model.lambda_reg * l1_norm
            
        # Apply L2 regularization
        elif model.reg_type == 'L2':
            l2_norm = sum( param.pow(2).sum() for param in model.parameters() )
            loss += model.lambda_reg * l2_norm

        # Save current loss value
        model.train_losses.append( loss.item() )

        # Zeroed gradients
        optimizer.zero_grad()
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Evaluate on validation set
        model.eval()
        with torch.no_grad():   
            y_valid_logits = model(X_valid).squeeze() # logits for valid data
            # y_valid_probs = torch.softmax(y_valid_logits, dim=1)
            # y_valid_pred = y_valid_probs.argmax(dim=1) # probabilities -> labels (predictions)
            valid_loss = model.loss_fn(y_valid_logits, y_valid)
            # Save current valid loss
            model.valid_losses.append( valid_loss.item() ) 

        # Early stopping 
        if valid_loss < best_valid_loss:
            # Saving current best model parameters and as the training finishes we will return these parameters
            # for the latest best model
            best_model_state = copy.deepcopy(model.state_dict())
            best_model = copy.deepcopy(model)
            best_model.load_state_dict(best_model_state)
            best_train_loss = loss.item()
            best_valid_loss = valid_loss.item()
            model.best_valid_loss = best_valid_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch} ...")
                print(f"\nEpoch: {epoch} | Train loss: {loss:.5f} | Valid loss: {valid_loss:.5f}")    
                print(f"Best model obtained at epoch {epoch - patience}")
                print(f"Epoch: {epoch - patience} | Train loss: {best_train_loss:.5f} | Valid loss: {best_valid_loss:.5f}")
                model.break_epoch = epoch
                break
        
        if (epoch) % (epochs/10) == 0:
            print(f"Epoch: {epoch} | Train loss: {loss:.5f} | Valid loss: {valid_loss:.5f}")
            model.break_epoch = epoch

    # We return best model and last model (model), last model is needed for plotting training process (losses over epochs)
    return best_model, model


"""Wrapper class for PyTorch Neural Network to be compatible with scikit-learn framework"""
class NeuralNetClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, hidden_layers_sizes=[64, 32], dropout_prob=0.2, regularization_type=None, 
                 lambda_reg=0.1, epochs=1000, lr=0.001, patience=100, random_state=68, device='cpu'):
        self.hidden_layers_sizes = hidden_layers_sizes
        self.dropout_prob = dropout_prob
        self.regularization_type = regularization_type
        self.lambda_reg = lambda_reg
        self.epochs = epochs
        self.lr = lr
        self.patience = patience
        self._best_model = None
        self._last_model = None
        self.random_state = random_state
        self.device = device

    @property
    def best_model(self):
        return self._best_model
    
    @best_model.setter
    def best_model(self, value):
        self._best_model = value

    @property
    def last_model(self):
        return self._last_model
    
    @last_model.setter
    def last_model(self, value):
        self._last_model = value

    """We pass X and y for training and validation as dataframes"""
    def fit(self, X_train, y_train, X_valid, y_valid):
        input_size = X_train.shape[1]
        model = NNForClassification(
                    input_size=input_size,
                    hidden_layers_sizes=self.hidden_layers_sizes,
                    n_classes=len( y_train.unique() ),
                    dropout_prob=self.dropout_prob,
                    regularization_type=self.regularization_type,
                    lambda_reg=self.lambda_reg
        ).to( self.device )


        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        
        # Converting dataframes to tensors
        X_train_tensor = torch.tensor( X_train.to_numpy(), dtype=torch.float32, device=self.device )
        y_train_tensor = torch.tensor( y_train.to_numpy(), dtype=torch.long, device=self.device )
        X_valid_tensor = torch.tensor( X_valid.to_numpy(), dtype=torch.float32, device=self.device )
        y_valid_tensor = torch.tensor( y_valid.to_numpy(), dtype=torch.long, device=self.device )
        
        # Training the model (returning best and last model)
        self.best_model, self.last_model = nn_train(
                                                model=model,
                                                optimizer=optimizer,
                                                X_train=X_train_tensor,
                                                y_train=y_train_tensor,
                                                X_valid=X_valid_tensor,
                                                y_valid=y_valid_tensor,
                                                epochs=self.epochs,
                                                patience=self.patience,
                                                random_state=self.random_state
        )
        return self
    
    def predict_proba(self, X):
        self.best_model.eval()
        with torch.inference_mode():
            X_tensor = torch.tensor(X.to_numpy(), dtype=torch.float32).to( self.device )
            logits = self.best_model(X_tensor).squeeze()
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs
    
    def predict(self, X):
        self.best_model.eval()
        with torch.inference_mode():
            X_tensor = torch.tensor(X.to_numpy(), dtype=torch.float32).to( self.device )
            logits = self.best_model(X_tensor).squeeze()
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1).cpu().numpy()
        return preds