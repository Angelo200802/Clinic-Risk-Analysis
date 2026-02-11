class Pattern {
  bool progRespFailure;
  bool dynamicSepsis;
  bool hemoDeterioration;

  Pattern({
    required this.progRespFailure,
    required this.dynamicSepsis,
    required this.hemoDeterioration,
  });

  factory Pattern.fromJson(Map<String, dynamic> json) {
    return Pattern(
      progRespFailure: json['progressive_resp_failure_pattern'] == 1,
      dynamicSepsis: json['dynamic_sepsis_pattern'] == 1,
      hemoDeterioration: json['progressive_hemo_deterioration'] == 1,
    );
  }
}
