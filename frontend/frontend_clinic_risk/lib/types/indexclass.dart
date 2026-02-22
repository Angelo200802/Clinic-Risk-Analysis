class CalculatedIndex {
  double shockIndex;
  double modifiedShockIndex;
  double ageIndex;
  double diastolicShockIndex;
  double ratePp;
  double ppIndex;

  CalculatedIndex({
    required this.shockIndex,
    required this.modifiedShockIndex,
    required this.ageIndex,
    required this.diastolicShockIndex,
    required this.ratePp,
    required this.ppIndex,
  });

  factory CalculatedIndex.fromJson(Map<String, dynamic> json) {
    return CalculatedIndex(
      shockIndex: json['shock_index'],
      modifiedShockIndex: json['modified_shock_index'],
      ageIndex: json['age_index'],
      diastolicShockIndex: json['diastolic_shock_index'],
      ratePp: json['rate_pp'],
      ppIndex: json['pp_index'],
    );
  }
}
