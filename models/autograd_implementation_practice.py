import torch
import torch.nn as nn
from torch.autograd import Function

# x = torch.randn(4, 3, device = 'cuda', requires_grad=True)
# y_prime = x * 2
# y = torch.randn(4, 3, device = 'cuda')

# print(y_prime)
# print(y_prime.backward(x))
# print(x.grad, x)

class EXP(Function):
    
    @staticmethod
    def forward(ctx, i):
        # v = torch.tensor(1.0, dtype=torch.float32, requires_grad=True)
        # ctx.save_for_backward(v)
        result = i.exp() # e^i
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        result, = ctx.saved_tensors
        print(grad_output, result)
        return result * grad_output

# x = torch.tensor(2.0, dtype=torch.float32, requires_grad=True)
# print(x)
# out = EXP.apply(x)
# print(out)
# print(out.backward())

class OPS(Function):
    @staticmethod
    def forward(ctx, x):
        y = 3 * x * x
        print(y)
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        print(grad_output)
        y, = ctx.saved_tensors
        return grad_output * y

x = torch.tensor(2.0, requires_grad=True)
print(x)
out = OPS.apply(x)
out.backward() 
print(out.grad)

# x = torch.tensor(5.0, requires_grad=True)
# y = 3 * x * x
# y.backward()
# print(x.grad)

# Inherit from Function
class LinearFunction(Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weight, bias=None):
        ctx.save_for_backward(input, weight, bias)
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        print(grad_output)
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias

x = torch.randn(3, 2, requires_grad=True)
w = torch.randn(3, 2, requires_grad=True)
y = LinearFunction.apply(x, w)
print(x, '\n', w.t(), '\n', y)
# y.backward(torch.ones((3, 3)))
y.backward(torch.randn(3, 3))
print(x.grad, w.grad)

# m = nn.Linear(20, 30, )
# input = torch.randn(128, 20, requires_grad=False)
# output = m(input)
# print(output.requires_grad)
# output.backward()

