from rest_framework import serializers

class HousePricePredictionSerializer(serializers.Serializer):
    city = serializers.CharField(max_length=255)
    current_price = serializers.FloatField()
    date_to_predict = serializers.DateField()

