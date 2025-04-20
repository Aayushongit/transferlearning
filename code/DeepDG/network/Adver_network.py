import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Function


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class GradientReversal(nn.Module):
    def __init__(self, alpha=1.0):
        super(GradientReversal, self).__init__()
        self.alpha = alpha
        
    def forward(self, x):
        return ReverseLayerF.apply(x, self.alpha)
    
    def set_alpha(self, alpha):
        self.alpha = alpha

class DomainDiscriminator(nn.Module):
    def __init__(self, input_dim=256, hidden_dims=[256, 128], num_domains=4, dropout=0.5,
                 use_spectral_norm=False, activation='relu'):
        super(DomainDiscriminator, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_domains = num_domains
        self.dropout_rate = dropout
        
        # Choose activation function
        if activation == 'relu':
            act_fn = nn.ReLU()
        elif activation == 'leaky_relu':
            act_fn = nn.LeakyReLU(0.2)
        elif activation == 'gelu':
            act_fn = nn.GELU()
        else:
            act_fn = nn.ReLU()
            
        # Build layers dynamically
        layers = []
        dims = [input_dim] + hidden_dims
        
        for i in range(len(dims)-1):
            if use_spectral_norm:
                layers.append(nn.utils.spectral_norm(nn.Linear(dims[i], dims[i+1])))
            else:
                layers.append(nn.Linear(dims[i], dims[i+1]))
                
            layers.append(nn.BatchNorm1d(dims[i+1]))
            layers.append(act_fn)
            layers.append(nn.Dropout(dropout))
            
        # Output layer
        if use_spectral_norm:
            layers.append(nn.utils.spectral_norm(nn.Linear(dims[-1], num_domains)))
        else:
            layers.append(nn.Linear(dims[-1], num_domains))
            
        self.layers = nn.Sequential(*layers)
        self.grl = GradientReversal()
        
    def forward(self, x, alpha=1.0):
        self.grl.set_alpha(alpha)
        features = self.grl(x)
        return self.layers(features)
    
    def get_parameters(self):
        return self.parameters()

class DomainAdaptationModel(nn.Module):
    def __init__(self, feature_extractor, classifier, domain_discriminator, 
                 use_entropy_weight=False, temperature=0.05):
        super(DomainAdaptationModel, self).__init__()
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.domain_discriminator = domain_discriminator
        self.use_entropy_weight = use_entropy_weight
        self.temperature = temperature
        
    def forward(self, x, alpha=1.0):
        features = self.feature_extractor(x)
        class_outputs = self.classifier(features)
        domain_outputs = self.domain_discriminator(features, alpha)
        
        return features, class_outputs, domain_outputs
    
    def get_entropy_weights(self, domain_outputs):
        """Calculate entropy-based weights for samples"""
        if not self.use_entropy_weight:
            return torch.ones_like(domain_outputs[:, 0])
            
        # Softmax with temperature
        probs = F.softmax(domain_outputs / self.temperature, dim=1)
        # Calculate entropy: -sum(p_i * log(p_i))
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
        # Normalize entropy to [0, 1]
        max_entropy = torch.log(torch.tensor(self.domain_discriminator.num_domains, 
                                              dtype=torch.float))
        normalized_entropy = entropy / max_entropy
        # Return weights based on entropy (higher entropy = more uncertain = higher weight)
        return normalized_entropy

def compute_domain_adaptation_loss(class_outputs, domain_outputs, class_labels, domain_labels, 
                                  model=None, class_weight=1.0, domain_weight=1.0):
    """
    Compute the combined loss for domain adaptation
    
    Args:
        class_outputs: Task classifier predictions
        domain_outputs: Domain discriminator predictions
        class_labels: Ground truth class labels
        domain_labels: Domain labels
        model: The full domain adaptation model (optional, for entropy weighting)
        class_weight: Weight for classification loss
        domain_weight: Weight for domain adversarial loss
    """
    # Task classification loss
    class_loss = F.cross_entropy(class_outputs, class_labels)
    
    # Domain classification loss with optional entropy weighting
    if model is not None and model.use_entropy_weight:
        weights = model.get_entropy_weights(domain_outputs)
        domain_loss = F.cross_entropy(domain_outputs, domain_labels, reduction='none')
        domain_loss = (domain_loss * weights).mean()
    else:
        domain_loss = F.cross_entropy(domain_outputs, domain_labels)
    
    # Combined loss
    total_loss = class_weight * class_loss + domain_weight * domain_loss
    
    return total_loss, class_loss, domain_loss

# Example usage
def adjust_alpha(epoch, max_epochs, gamma=10):
    """Progressive adjustment of gradient reversal strength"""
    p = epoch / max_epochs
    return 2.0 / (1.0 + np.exp(-gamma * p)) - 1.0
    
class Discriminator(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=256, num_domains=4):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_domains),
        ]
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
