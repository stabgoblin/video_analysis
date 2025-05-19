import yaml
from collections import defaultdict
from typing import List, Dict

class ActivityAnalyzer:
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize with security rules from config.yaml.
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        # Initialize activity log
        self.activity_log = defaultdict(int)
        
        # Preload security settings
        self.suspicious_tags = list(self.config["security_tags"].values())
        self.alert_thresholds = self.config["alert_thresholds"]

    def update_log(self, caption: str) -> None:
        """
        Update counts of suspicious terms in the activity log.
        
        Args:
            caption: Generated caption from BLIP-2 (with security tags)
        """
        for tag in self.suspicious_tags:
            if tag in caption:
                self.activity_log[tag] += 1

    def check_alerts(self) -> List[str]:
        """
        Check if any activity exceeds configured thresholds.
        
        Returns:
            List of alert messages (empty if no alerts)
        """
        alerts = []
        for tag, threshold in self.alert_thresholds.items():
            if self.activity_log[tag] >= threshold:
                alerts.append(
                    f"{tag} detected {self.activity_log[tag]} times "
                    f"(threshold: {threshold})"
                )
        return alerts

    def reset_log(self) -> None:
        """Clear the activity log (e.g., at start of new hour)."""
        self.activity_log.clear()


# Example usage
if __name__ == "__main__":
    analyzer = ActivityAnalyzer()
    
    # Simulate processing
    test_captions = [
        "**PERSON** near entrance",
        "**PERSON** with **UNATTENDED_ITEM**",
        "**PERSON** loitering at **NIGHT_ACTIVITY**"
    ]
    
    for caption in test_captions:
        analyzer.update_log(caption)
        print(f"Processed: {caption}")
    
    print("\nAlerts:")
    print("\n".join(analyzer.check_alerts()) or "No alerts")