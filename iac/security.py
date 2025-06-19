from typing import Dict, Optional
import pulumi
import pulumi_alicloud as alicloud
import config


def create_security_group(vpc_id: pulumi.Input[str]) -> alicloud.ecs.SecurityGroup:
    """
    Create a security group for the Freqtrade ECS instance.
    
    Args:
        vpc_id: The ID of the VPC where the security group will be created
        
    Returns:
        alicloud.ecs.SecurityGroup: The created security group resource
    """
    return alicloud.ecs.SecurityGroup(
        config.config.security_group_name,
        vpc_id=vpc_id,
        name=config.config.security_group_name,
        description="Security group for freqtrade bot",
        tags={
            "Name": config.config.security_group_name,
            "Environment": pulumi.get_stack(),
            "ManagedBy": "Pulumi",
            "Project": "freqtrade"
        },
    )


def create_security_group_rules(
    security_group_id: pulumi.Input[str]
) -> Dict[str, alicloud.ecs.SecurityGroupRule]:
    """
    Create security group rules for the Freqtrade ECS instance.
    
    Args:
        security_group_id: The ID of the security group
        
    Returns:
        Dict[str, alicloud.ecs.SecurityGroupRule]: Dictionary of security group rules
    """
    rules = {}
    
    # SSH access (Warning: This is open to the internet!)
    rules["ssh"] = alicloud.ecs.SecurityGroupRule(
        f"{config.config.security_group_name}-ssh-rule",
        type="ingress",
        ip_protocol="tcp",
        port_range="22/22",
        cidr_ip="0.0.0.0/0",  # Warning: This allows SSH from any IP address
        security_group_id=security_group_id,
        description="Allow SSH access from anywhere",
    )

    # Freqtrade API access
    rules["api"] = alicloud.ecs.SecurityGroupRule(
        f"{config.config.security_group_name}-api-rule",
        type="ingress",
        ip_protocol="tcp",
        port_range=f"{config.config.api_port}/{config.config.api_port}",
        cidr_ip="0.0.0.0/0",  # Warning: This allows API access from any IP address
        security_group_id=security_group_id,
        description=f"Allow Freqtrade API access on port {config.config.api_port}",
    )
    
    return rules


def get_security_outputs(
    security_group: alicloud.ecs.SecurityGroup,
    rules: Dict[str, alicloud.ecs.SecurityGroupRule]
) -> Dict[str, pulumi.Output[str]]:
    """
    Get the security group outputs for export.
    
    Args:
        security_group: The security group resource
        rules: Dictionary of security group rules
        
    Returns:
        Dict[str, pulumi.Output[str]]: Dictionary of security group outputs
    """
    outputs = {
        "security_group_id": security_group.id,
        "security_group_name": security_group.security_group_name,
    }
    
    # Add rule IDs to outputs
    for rule_name, rule in rules.items():
        outputs[f"rule_{rule_name}_id"] = rule.id
    
    return outputs


# Export security resources if this module is run directly
if __name__ == "__main__":
    # This is just for testing - in practice, VPC ID should be passed as a parameter
    vpc = alicloud.vpc.Network.get("test-vpc", id="vpc-test-id")
    sg = create_security_group(vpc.id)
    rules = create_security_group_rules(sg.id)
    outputs = get_security_outputs(sg, rules)
    for key, value in outputs.items():
        pulumi.export(f"security_{key}", value)
