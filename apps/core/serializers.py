from rest_framework import serializers

class HousePricePredictionSerializer(serializers.Serializer):
    current_price = serializers.DecimalField(max_digits=12, decimal_places=2)
    date_to_predict = serializers.DateField()
