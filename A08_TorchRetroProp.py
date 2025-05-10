#=================================
# Gradientes simples con Pythorch
#=================================
# Chan Campos Ashanty Iyari
# Fundamentos de IA
# ESFM IPN Marzo 2025
#=================================
import torch

x = torch.tensor(1.0)
y = torch.tensor(2.0)

#=========================================================
# Variable de diferenciación (s/dw) -> requires_grad=True
#=========================================================
w = torch.tensor(1.0, requires_grad=True)

#=============================
# Evaluación cálculo de costo
#=============================
y_predicted = w * x
loss = (y_predicted - y)**2

#==========================================
# retropropagación para calcular gradiente
# w.grad es el gradiente 
#==========================================
loss.backward()
print(w.grad)

#=============================================
# NUevos coeficientes (descenso de gradiente)
# repetir evaluación y retropropagación 
#=============================================
with torch.no_grad():
    w -= 0.01 * w.grad
w.grad.zero()
print(w)