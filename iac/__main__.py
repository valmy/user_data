"""
Main entry point for Pulumi infrastructure as code.

This module orchestrates the creation of all resources needed for the Freqtrade bot
on Alibaba Cloud, including VPC, security groups, ECS instances, and EIPs.
"""
import pulumi

# Import local modules directly
import config as cfg
import vpc
import security
import compute

# Configuration
USE_EIP = True  # Set to False if you don't need a static IP

# Create VPC
vpc_resource = vpc.create_vpc()

# Create VSwitch in the Jakarta region
vswitch = vpc.create_vswitch(
    vpc_id=vpc_resource.id,
    zone_id=cfg.config.zone_id
)

# Create security group and rules
security_group = security.create_security_group(vpc_id=vpc_resource.id)
security_rules = security.create_security_group_rules(security_group_id=security_group.id)

# Create ECS instance with VPC and VSwitch
ec2_instance, eip = compute.create_ecs_instance(
    security_group_ids=[security_group.id],
    vswitch_id=vswitch.id,
    use_eip=USE_EIP
)

# Get outputs from all modules
vpc_outputs = vpc.get_vpc_outputs(vpc_resource, vswitch)
security_outputs = security.get_security_outputs(security_group, security_rules)
compute_outputs = compute.get_compute_outputs(ec2_instance, eip)

# Export all outputs
for prefix, outputs in [
    ("vpc_", vpc_outputs),
    ("security_", security_outputs),
    ("", compute_outputs),  # No prefix for compute outputs as they're commonly used
]:
    for key, value in outputs.items():
        pulumi.export(f"{prefix}{key}", value)

# Add a warning if using default SSH access
if cfg.config.ssh_source_ip == "0.0.0.0/0":
    pulumi.log.warn(
        "WARNING: SSH access is open to the internet (0.0.0.0/0). "
        "For production environments, restrict access to specific IP addresses."
    )

# Add a note about EIP charges if EIP is enabled
if USE_EIP:
    pulumi.log.info(
        "NOTE: An Elastic IP (EIP) has been created. "
        "Please note that EIPs may incur charges when not associated with a running instance."
    )
