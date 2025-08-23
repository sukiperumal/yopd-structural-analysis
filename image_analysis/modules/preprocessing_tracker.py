#!/usr/bin/env python3
"""
Preprocessing step tracking module for enhanced image quality assessment
"""

import pandas as pd
from typing import Dict, List, Tuple, Any

class PreprocessingStepTracker:
    """Track and document all preprocessing steps performed"""
    
    def __init__(self):
        self.steps = []
        self.technical_details = {}
    
    def log_step(self, step_name: str, method: str, parameters: Dict, 
                 input_shape: Tuple, output_shape: Tuple, success: bool = True):
        """Log a preprocessing step with technical details"""
        step_info = {
            'step_name': step_name,
            'method': method,
            'parameters': parameters,
            'input_shape': input_shape,
            'output_shape': output_shape,
            'success': success,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        self.steps.append(step_info)
        print(f"✓ Logged: {step_name} using {method} - {'SUCCESS' if success else 'FAILED'}")
    
    def get_report(self) -> Dict:
        """Get comprehensive preprocessing report"""
        total_steps = len(self.steps)
        successful_steps = sum(1 for step in self.steps if step['success'])
        failed_steps = total_steps - successful_steps
        
        return {
            'total_steps': total_steps,
            'successful_steps': successful_steps,
            'failed_steps': failed_steps,
            'success_rate': successful_steps / total_steps if total_steps > 0 else 0.0,
            'step_details': self.steps
        }
    
    def print_summary(self):
        """Print a summary of tracked steps"""
        report = self.get_report()
        print(f"\nPreprocessing Summary:")
        print(f"  Total steps: {report['total_steps']}")
        print(f"  Successful: {report['successful_steps']}")
        print(f"  Failed: {report['failed_steps']}")
        print(f"  Success rate: {report['success_rate']:.1%}")
        
        if report['failed_steps'] > 0:
            print("\nFailed steps:")
            for step in self.steps:
                if not step['success']:
                    print(f"  - {step['step_name']}: {step.get('parameters', {}).get('error', 'Unknown error')}")
    
    def get_step_by_name(self, step_name: str) -> Dict:
        """Get details for a specific step"""
        for step in self.steps:
            if step['step_name'] == step_name:
                return step
        return {}
    
    def clear(self):
        """Clear all logged steps"""
        self.steps = []
        self.technical_details = {}
        print("Cleared all preprocessing step logs")
