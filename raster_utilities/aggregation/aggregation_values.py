class AggregationModes:
    Continuous, Categorical = range(2)
class AggregationTypes:
    Resolution, Factor, Size = range(3)
class ContinuousAggregationStats:
    Min, Max, Range, Sum, Mean, SD, Count = range(7)
    All = [Min, Max, Range, Sum, Mean, SD, Count]
    Names = {0:"min", 1:"max", 2:"range", 3:"sum", 4:"mean", 5:"sd", 6:"count"}

class CategoricalAggregationStats:
    Majority, Fractions, LikeAdjacencies = range(3)
    All = [Majority, Fractions, LikeAdjacencies]
    Names = {0:"majority", 1:"fractions", 2:"like-adjacencies"}