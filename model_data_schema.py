from pymodm import MongoModel, fields


class ModelData(MongoModel):
    user = fields.CharField(required=True)
    graph = fields.CharField(required=True)
    count = fields.IntegerField(required=True)
    overallEvents = fields.DictField(required=True)
    eventsPerDay = fields.DictField(required=True)
    published_date = fields.DateTimeField(required=True)

    class Meta:
        collection_name = 'modelData'

    @classmethod
    def add_data(cls, user, graph, count, overall_events, events_per_day, published_date):
        try:
            cls.objects.raw({'user': user}).first()

        except cls.DoesNotExist:
            data = cls(user=user, graph=graph, count=count, overallEvents=overall_events,
                       eventsPerDay=events_per_day, published_date=published_date)
            data.save()

    @classmethod
    def check_user(cls, user):
        check_user = False
        try:
            username = cls.objects.raw({'user': user}).first()
            if username:
                print(f"User '{user}' already exists in the database")
                check_user = True

            return check_user

        except cls.DoesNotExist:
            return check_user
