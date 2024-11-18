from django.shortcuts import render
#from .forms import HousePriceForm
#from .arima_model import load_trained_model, predict_price

# Create the view to handle form submission and make predictions
'''
def predict_house_price(request):
    if request.method == 'POST':
        form = HousePriceForm(request.POST)
        if form.is_valid():
            current_price = form.cleaned_data['current_price']
            model = load_trained_model()  # Load the pre-trained model
            predicted_price = predict_price(model, current_price)
            return render(request, 'predict_price.html', {'form': form, 'predicted_price': predicted_price})
    else:
        form = HousePriceForm()
    return render(request, 'predict_price.html', {'form': form})
'''