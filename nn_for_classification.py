import torch
import copy
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

"""Class for Neural Network Classifier architecture"""
class MLPClassifier(torch.nn.Module):
    def __init__(
            self, 
            input_size, 
            hidden_layers_sizes=[64, 32, 16], 
            n_classes=2, 
            dropout_prob=0.2
        ):
        super().__init__()
        self.n_classes = n_classes

        layers = [torch.nn.Linear(in_features=input_size, out_features=hidden_layers_sizes[0])]
        for i in range( len(hidden_layers_sizes) - 1 ):
            layers.extend([
                torch.nn.BatchNorm1d(num_features=hidden_layers_sizes[i]),
                torch.nn.ReLU(),
                torch.nn.Dropout(p=dropout_prob),
                torch.nn.Linear(in_features=hidden_layers_sizes[i], out_features=hidden_layers_sizes[i+1])
            ])
        layers.extend([
            torch.nn.BatchNorm1d(num_features=hidden_layers_sizes[-1]),
            torch.nn.ReLU(), 
            torch.nn.Dropout(p=dropout_prob), 
            torch.nn.Linear(in_features=hidden_layers_sizes[-1], out_features=n_classes) 
        ])

        self.layer_stack = torch.nn.Sequential( *layers )
        # self.output_activation = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.layer_stack(x)
        # x = self.output_activation(x)
        return x

class ModelTrainer():
    def __init__(
            self, 
            model, 
            X_train, 
            y_train, 
            X_valid, 
            y_valid, 
            class_weights=None, 
            learning_rate=0.01,
            optimizer=None, 
            epochs=1000, 
            patience=100, 
            regularization_type=None, 
            lambda_reg=0.1, 
            random_state=68
        ):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_valid = X_valid
        self.y_valid = y_valid
        self.class_weights = class_weights
        self.loss_fn = torch.nn.CrossEntropyLoss( weight=self.class_weights )
        self.learning_rate = learning_rate
        # L2 regularization is implemented as weight decay in the optimizer
        self.weight_decay = lambda_reg if (
            regularization_type is not None 
            and regularization_type.lower() == 'l2'
        ) else 0.0 
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.learning_rate, 
            weight_decay=self.weight_decay
        ) if optimizer is None else optimizer
        self.epochs = epochs
        self.patience = patience
        self.reg_type = regularization_type
        self.lambda_reg = lambda_reg
        self.random_state = random_state
        self.train_losses = []
        self.valid_losses = []
        self.best_model = None
        self.best_train_loss = float('inf')
        self.best_valid_loss = float('inf')
        self.break_epoch = 1
        self.best_epoch = 1

    """Neural Network training process"""
    def train(self):
        torch.manual_seed( self.random_state )
        epochs_no_improve = 0

        for epoch in range(1, self.epochs+1):
            self.model.train()
            
            # Forward pass
            y_train_logits = self.model( self.X_train ) # logits for train data
            # y_train_probs = torch.softmax(y_train_logits, dim=1)
            
            # Calculate loss (Inside CrossEntropy there is Softmax calculated so as input we put logits)
            loss = self.loss_fn(y_train_logits, self.y_train)

            # Apply L1 regularization
            if (self.reg_type is not None and self.reg_type.lower() == 'l1'):
                l1_norm = sum(param.abs().sum() for param in self.model.parameters())
                loss += self.lambda_reg * l1_norm
                
            # Apply L2 regularization, previous implementation (commented out) was based on manual calculation of L2 norm, 
            # but now we use weight decay in the optimizer which is more efficient and numerically stable
            # elif self.reg_type.lower() == 'l2':
            #     l2_norm = sum(param.pow(2).sum() for param in self.model.parameters())
            #     loss += self.lambda_reg * l2_norm

            # Save current loss value
            self.train_losses.append( loss.item() )

            # Zeroed gradients
            self.optimizer.zero_grad()
            
            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()

            # Evaluate on validation set
            self.model.eval()
            with torch.no_grad():   
                y_valid_logits = self.model( self.X_valid ) # logits for valid data
                # y_valid_probs = torch.softmax(y_valid_logits, dim=-1)
                # y_valid_pred = y_valid_probs.argmax(dim=1) # probabilities -> labels (predictions)
                valid_loss = self.loss_fn(y_valid_logits, self.y_valid)
                # Save current valid loss
                self.valid_losses.append( valid_loss.item() ) 

            # Early stopping 
            if valid_loss < self.best_valid_loss:
                # Saving current best model parameters and as the training finishes we will return these parameters
                # for the latest best model
                best_model_state = copy.deepcopy(self.model.state_dict())
                self.best_model = copy.deepcopy(self.model)
                self.best_model.load_state_dict(best_model_state)
                self.best_train_loss = loss.item()
                self.best_valid_loss = valid_loss.item()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    self.best_epoch = int(epoch - self.patience)
                    print(f"Early stopping at epoch {epoch} ...")
                    print(f"\nEpoch: {epoch} | Train loss: {loss:.5f} | Valid loss: {valid_loss:.5f}")    
                    print(f"Best model obtained at epoch {self.best_epoch}")
                    print(f"Epoch: {self.best_epoch} | Train loss: {self.best_train_loss:.5f} | Valid loss: {self.best_valid_loss:.5f}")
                    self.break_epoch = epoch
                    break
            
            if (epoch) % 100 == 0:
                print(f"Epoch: {epoch} | Train loss: {loss:.5f} | Valid loss: {valid_loss:.5f}")
                self.break_epoch = epoch

        # We return self (finished training process)
        return self

from sklearn.model_selection import train_test_split

"""Wrapper class for PyTorch Neural Network to be compatible with scikit-learn framework"""
class NeuralNetClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
            self, 
            hidden_layers_sizes=[64, 32], 
            dropout_prob=0.2, 
            class_weights=None, 
            learning_rate=0.001, 
            optimizer=None,
            validation_split=0.2, 
            regularization_type=None, 
            lambda_reg=0.1, 
            epochs=1000, 
            patience=100, 
            random_state=68, 
            device='cpu'
        ):
        self.hidden_layers_sizes = hidden_layers_sizes
        self.dropout_prob = dropout_prob
        self.class_weights = class_weights
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.validation_split = validation_split
        self.epochs = epochs
        self.patience = patience
        self.regularization_type = regularization_type
        self.lambda_reg = lambda_reg
        self.random_state = random_state
        self.device = device
        self.history = None

    """We pass X and y for training and validation as dataframes"""
    def fit(self, X, y): #, X_valid, y_valid):
        input_size = X.shape[1]
        model = MLPClassifier(
            input_size=input_size,
            hidden_layers_sizes=self.hidden_layers_sizes,
            n_classes=len( y.unique() ),
            dropout_prob=self.dropout_prob
        ).to( self.device )

        # optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        # self.optimizer = optimizer
        
        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y, 
            test_size=self.validation_split, 
            random_state=self.random_state, 
            stratify=y
        )
        # Converting dataframes to tensors
        X_train_tensor = torch.tensor( X_train.to_numpy(), dtype=torch.float32, device=self.device )
        y_train_tensor = torch.tensor( y_train.to_numpy(), dtype=torch.long, device=self.device )
        X_valid_tensor = torch.tensor( X_valid.to_numpy(), dtype=torch.float32, device=self.device )
        y_valid_tensor = torch.tensor( y_valid.to_numpy(), dtype=torch.long, device=self.device )
        
        # Training the model and saving the history (so the ModelTrainer instance with all the training information and best model parameters)
        trainer = ModelTrainer(
            model=model,
            X_train=X_train_tensor,
            y_train=y_train_tensor,
            X_valid=X_valid_tensor,
            y_valid=y_valid_tensor,
            class_weights=self.class_weights,
            learning_rate=self.learning_rate,
            optimizer=self.optimizer,
            epochs=self.epochs,
            patience=self.patience,
            regularization_type=self.regularization_type,
            lambda_reg=self.lambda_reg,
            random_state=self.random_state
        )
        trainer.train()

        self.history = trainer

        return self
    
    def predict_proba(self, X):
        self.history.best_model.eval()
        with torch.inference_mode():
            X_tensor = torch.tensor(X.to_numpy(), dtype=torch.float32).to( self.device )
            logits = self.history.best_model(X_tensor) 
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        return probs
    
    def predict(self, X):
        self.history.best_model.eval()
        with torch.inference_mode():
            X_tensor = torch.tensor(X.to_numpy(), dtype=torch.float32).to( self.device )
            logits = self.history.best_model(X_tensor) 
            probs = torch.softmax(logits, dim=-1)
            preds = probs.argmax(dim=1).cpu().numpy()
        return preds