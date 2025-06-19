from typing import Optional
from dataclasses import dataclass
import pulumi


@dataclass
class Config:
    """Pulumi configuration manager for Freqtrade infrastructure."""
    
    # Network configuration
    region: str = 'ap-southeast-5'  # Jakarta region
    vpc_name: str = 'freqtrade-vpc'
    vpc_cidr: str = '10.0.0.0/16'
    vswitch_cidr: str = '10.0.1.0/24'  # Subnet within VPC CIDR
    zone_id: str = 'ap-southeast-5a'  # Zone ID in Jakarta region
    
    # Instance configuration
    instance_type: str = 'ecs.e-c1m2.large'
    image_id: str = 'ubuntu_24_04_x64_20G_alibase_20250527.vhd'
    ecs_instance_name: str = 'freqtrade-bot'
    key_name: str = 'wabisabi'  # SSH key pair name
    
    # Security configuration
    security_group_name: str = 'freqtrade-sg'
    ssh_source_ip: str = '0.0.0.0/0'  # Warning: This allows SSH from any IP
    
    # Application configuration
    api_port: int = 8080  # Default API port for Freqtrade
    
    # SSH configuration
    ssh_username: str = 'root'  # Default SSH username for Ubuntu
    ssh_private_key_path: str = '~/.ssh/id_ed25519'  # Default SSH private key path
    
    # Storage configuration
    system_disk_category: str = 'cloud_auto'
    system_disk_size: int = 40  # GB
    
    # Network bandwidth
    internet_max_bandwidth_out: int = 1  # Mbps
    
    # VSwitch settings
    vswitch_name: str = 'freqtrade-vswitch'
    vswitch_description: str = 'VSwitch for Freqtrade'

    def __init__(self):
        # Initialize any instance-specific attributes here
        # Set zone_id based on region if not specified
        if not hasattr(self, 'zone_id') or not self.zone_id:
            if self.region == 'ap-southeast-5':
                self.zone_id = 'ap-southeast-5a'  # Default zone for Jakarta

    @classmethod
    def from_pulumi_config(cls) -> 'Config':
        """Load configuration from Pulumi config."""
        config = pulumi.Config()
        
        # Create a new instance with default values
        cfg = cls()
        
        # Update with values from Pulumi config if they exist
        for field in cfg.__dataclass_fields__:
            if config.get(field) is not None:
                value = config.get(field)
                # Convert string values to appropriate types
                if field in ['api_port', 'system_disk_size', 'internet_max_bandwidth_out']:
                    value = int(value)
                setattr(cfg, field, value)
        
        return cfg

    def validate(self) -> None:
        """Validate the configuration."""
        if not self.ssh_source_ip or self.ssh_source_ip == '0.0.0.0/0':
            pulumi.log.warn(
                "WARNING: SSH access is open to the internet (0.0.0.0/0). "
                "Consider restricting access to specific IP addresses for better security."
            )


# Export config for easy access
config = Config.from_pulumi_config()
config.validate()
