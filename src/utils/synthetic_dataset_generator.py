import pandas as pd
import random
from typing import List, Tuple, Dict
import json

class SyntheticRequirementsDatasetGenerator:
    """
    Generate high-quality synthetic requirements-to-activities dataset
    following NASA/IEEE standards for requirements writing.
    """
    
    def __init__(self):
        # Modal verbs for different requirement types
        self.modals = {
            'mandatory': ['shall', 'must'],
            'desirable': ['should'],
            'optional': ['may', 'can']
        }
        
        # Requirement patterns based on IEEE 830 standard
        self.requirement_patterns = {
            'functional': [
                "The {system} shall {action} {object} {constraint}",
                "The {system} shall {action} {object} when {condition}",
                "The {system} shall provide {capability} to {user}",
                "The {system} shall {action} within {timing} of {trigger}"
            ],
            'performance': [
                "The {system} shall {action} {object} in less than {time}",
                "The {system} shall process {quantity} {object} per {timeunit}",
                "The {system} shall maintain {metric} below {threshold}",
                "The {system} shall achieve {accuracy}% accuracy for {operation}"
            ],
            'interface': [
                "The {system} shall interface with {external_system} using {protocol}",
                "The {system} shall accept {input_type} in {format} format",
                "The {system} shall provide {output_type} via {interface}",
                "The {system} shall communicate with {component} at {rate} Hz"
            ],
            'reliability': [
                "The {system} shall operate continuously for {duration} hours",
                "The {system} shall achieve {percentage}% uptime",
                "The {system} shall recover from {failure_type} within {time}",
                "The {system} shall maintain {metric} with {percentage}% reliability"
            ]
        }
        
        # Activity patterns following work breakdown structure
        self.activity_patterns = {
            'design': [
                "Design {component} {subsystem}",
                "Define {interface} specifications",
                "Create {artifact} architecture",
                "Develop {component} design document"
            ],
            'implement': [
                "Implement {function} module",
                "Code {algorithm} for {operation}",
                "Develop {component} software",
                "Build {subsystem} components"
            ],
            'test': [
                "Test {component} {test_type}",
                "Verify {requirement} compliance",
                "Validate {function} performance",
                "Execute {test_type} test suite"
            ],
            'monitor': [
                "Monitor {metric} levels",
                "Track {parameter} status",
                "Log {event} occurrences",
                "Record {measurement} data"
            ]
        }
        
        # Domain vocabularies for different systems
        self.domains = {
            'spacecraft': {
                'systems': ['spacecraft', 'satellite', 'probe', 'vehicle'],
                'components': ['propulsion system', 'attitude control', 'power system', 
                              'thermal control', 'communication system', 'payload'],
                'actions': ['control', 'maintain', 'monitor', 'transmit', 'receive', 
                           'process', 'store', 'execute', 'calculate', 'adjust'],
                'metrics': ['temperature', 'pressure', 'voltage', 'orientation', 
                           'velocity', 'altitude', 'signal strength', 'power consumption'],
                'constraints': ['±1°C', '±0.1 degrees', '<100ms', '>95%', '<10W']
            },
            'medical': {
                'systems': ['medical device', 'patient monitor', 'diagnostic system', 'analyzer'],
                'components': ['sensor array', 'display unit', 'alarm system', 
                              'data processor', 'user interface', 'safety mechanism'],
                'actions': ['measure', 'display', 'alert', 'record', 'analyze', 
                           'calibrate', 'diagnose', 'report', 'validate', 'notify'],
                'metrics': ['heart rate', 'blood pressure', 'oxygen level', 'temperature',
                           'response time', 'accuracy', 'sensitivity', 'specificity'],
                'constraints': ['±2%', '<1 second', '>99%', '±0.5°C', '<5% error']
            },
            'automotive': {
                'systems': ['vehicle control system', 'ADAS', 'ECU', 'infotainment system'],
                'components': ['braking module', 'steering controller', 'sensor fusion',
                              'navigation unit', 'safety controller', 'CAN interface'],
                'actions': ['detect', 'brake', 'steer', 'accelerate', 'warn',
                           'navigate', 'communicate', 'diagnose', 'actuate', 'prevent'],
                'metrics': ['speed', 'distance', 'braking force', 'response time',
                           'detection range', 'accuracy', 'latency', 'throughput'],
                'constraints': ['<200ms', '>90%', '<5m', '±10%', '>99.9%']
            }
        }
    
    def generate_requirement(self, domain: str, req_type: str, req_id: str) -> Dict:
        """Generate a single high-quality requirement."""
        pattern = random.choice(self.requirement_patterns[req_type])
        modal = random.choice(self.modals['mandatory'])
        
        vocab = self.domains[domain]
        
        # Fill in the pattern
        requirement = pattern.format(
            system=random.choice(vocab['systems']),
            action=random.choice(vocab['actions']),
            object=random.choice(vocab['components']),
            capability=f"{random.choice(vocab['actions'])} {random.choice(vocab['metrics'])}",
            user='operator',
            constraint=random.choice(vocab['constraints']),
            condition=f"{random.choice(vocab['metrics'])} exceeds threshold",
            timing=random.choice(['100ms', '1 second', '500ms', '2 seconds']),
            trigger=f"{random.choice(vocab['metrics'])} change",
            time=random.choice(['100ms', '1 second', '5 seconds']),
            quantity=random.choice(['100', '1000', '50']),
            timeunit=random.choice(['second', 'minute', 'hour']),
            metric=random.choice(vocab['metrics']),
            threshold=random.choice(vocab['constraints']),
            accuracy=random.choice(['95', '99', '99.9']),
            operation=f"{random.choice(vocab['actions'])} operation",
            external_system=random.choice(['ground station', 'external sensor', 'data bus']),
            protocol=random.choice(['CAN', 'RS-232', 'Ethernet', 'SPI']),
            input_type=random.choice(['command', 'data', 'signal']),
            format=random.choice(['JSON', 'binary', 'XML', 'hexadecimal']),
            output_type=random.choice(['telemetry', 'status', 'measurement']),
            interface=random.choice(['API', 'serial port', 'network socket']),
            component=random.choice(vocab['components']),
            rate=random.choice(['10', '50', '100', '1000']),
            duration=random.choice(['24', '48', '168', '720']),
            percentage=random.choice(['99', '99.9', '99.99']),
            failure_type=random.choice(['power loss', 'communication failure', 'sensor fault'])
        )
        
        # Replace 'shall' with the chosen modal
        requirement = requirement.replace('shall', modal, 1)
        
        return {
            'ID': req_id,
            'Requirement Name': f"{req_type.title()} Requirement {req_id}",
            'Requirement Text': requirement,
            'Type': req_type,
            'Domain': domain
        }
    
    def generate_activity(self, activity_type: str, domain: str, activity_id: str) -> Dict:
        """Generate a single activity."""
        pattern = random.choice(self.activity_patterns[activity_type])
        vocab = self.domains[domain]
        
        activity = pattern.format(
            component=random.choice(vocab['components']),
            subsystem=random.choice(['module', 'subsystem', 'unit']),
            interface=random.choice(['API', 'protocol', 'bus']),
            artifact=random.choice(['software', 'hardware', 'system']),
            function=f"{random.choice(vocab['actions'])} {random.choice(vocab['metrics'])}",
            algorithm=random.choice(['control', 'filtering', 'processing', 'detection']),
            operation=random.choice(vocab['actions']),
            test_type=random.choice(['unit', 'integration', 'performance', 'stress']),
            requirement=f"REQ-{random.randint(100,999)}",
            metric=random.choice(vocab['metrics']),
            parameter=random.choice(vocab['metrics']),
            event=f"{random.choice(vocab['metrics'])} change",
            measurement=random.choice(vocab['metrics'])
        )
        
        return {
            'Activity ID': activity_id,
            'Activity Name': activity,
            'Type': activity_type,
            'Domain': domain
        }
    
    def generate_mapping_rules(self) -> Dict[str, List[str]]:
        """Define intelligent mapping rules between requirement types and activity types."""
        return {
            'functional': ['design', 'implement', 'test'],
            'performance': ['implement', 'test', 'monitor'],
            'interface': ['design', 'implement', 'test'],
            'reliability': ['implement', 'test', 'monitor']
        }
    
    def generate_dataset(self, num_requirements: int = 100, 
                        num_activities: int = 300,
                        domain: str = 'spacecraft') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Generate complete synthetic dataset with requirements, activities, and mappings."""
        
        # Generate requirements
        requirements = []
        req_types = list(self.requirement_patterns.keys())
        
        for i in range(num_requirements):
            req_type = random.choice(req_types)
            req_id = f"REQ-{i+1:03d}"
            requirements.append(self.generate_requirement(domain, req_type, req_id))
        
        # Generate activities
        activities = []
        activity_types = list(self.activity_patterns.keys())
        
        for i in range(num_activities):
            act_type = random.choice(activity_types)
            # Create hierarchical numbering
            major = (i // 50) + 1
            minor = (i // 10) % 5 + 1
            sub = i % 10 + 1
            activity_id = f"{major}.{minor}.{sub}"
            
            activity = self.generate_activity(act_type, domain, activity_id)
            activity['Activity Name'] = f"{activity_id} {activity['Activity Name']}"
            activities.append(activity)
        
        # Generate intelligent mappings
        mapping_rules = self.generate_mapping_rules()
        mappings = []
        
        for req in requirements:
            req_type = req['Type']
            compatible_activity_types = mapping_rules[req_type]
            
            # Find compatible activities
            compatible_activities = [
                act for act in activities 
                if act['Type'] in compatible_activity_types
            ]
            
            # Select 2-5 activities per requirement
            num_mappings = random.randint(2, 5)
            if len(compatible_activities) >= num_mappings:
                selected_activities = random.sample(compatible_activities, num_mappings)
                
                for act in selected_activities:
                    mappings.append({
                        'ID': req['ID'],
                        'Satisfied By': act['Activity Name'],
                        'Confidence': random.choice(['High', 'Medium', 'Low'])
                    })
        
        # Create dataframes
        req_df = pd.DataFrame(requirements)
        act_df = pd.DataFrame(activities)
        
        # Create ground truth in the expected format
        ground_truth_grouped = pd.DataFrame(mappings).groupby('ID')['Satisfied By'].apply(
            lambda x: ', '.join(x)
        ).reset_index()
        
        return req_df, act_df, ground_truth_grouped
    
    def save_dataset(self, req_df: pd.DataFrame, act_df: pd.DataFrame, 
                    ground_truth_df: pd.DataFrame, prefix: str = "synthetic"):
        """Save the generated dataset to CSV files."""
        req_df.to_csv(f"{prefix}_requirements.csv", index=False)
        act_df.to_csv(f"{prefix}_activities.csv", index=False)
        ground_truth_df.to_csv(f"{prefix}_manual_matches.csv", index=False)
        
        # Save metadata
        metadata = {
            'num_requirements': len(req_df),
            'num_activities': len(act_df),
            'num_mappings': len(ground_truth_df),
            'avg_activities_per_req': ground_truth_df['Satisfied By'].str.count(',').mean() + 1,
            'requirement_types': req_df['Type'].value_counts().to_dict(),
            'activity_types': act_df['Type'].value_counts().to_dict()
        }
        
        with open(f"{prefix}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Dataset saved with prefix: {prefix}")
        print(f"Requirements: {len(req_df)}")
        print(f"Activities: {len(act_df)}")
        print(f"Ground truth mappings: {len(ground_truth_df)}")


# Example usage
if __name__ == "__main__":
    generator = SyntheticRequirementsDatasetGenerator()
    
    # Generate datasets for different domains
    for domain in ['spacecraft', 'medical', 'automotive']:
        req_df, act_df, ground_truth = generator.generate_dataset(
            num_requirements=100,
            num_activities=300,
            domain=domain
        )
        
        generator.save_dataset(req_df, act_df, ground_truth, prefix=f"synthetic_{domain}")
        
        print(f"\nGenerated {domain} dataset:")
        print(f"Sample requirement: {req_df.iloc[0]['Requirement Text']}")
        print(f"Sample activity: {act_df.iloc[0]['Activity Name']}")
        print(f"Sample mapping: {ground_truth.iloc[0]['Satisfied By']}")