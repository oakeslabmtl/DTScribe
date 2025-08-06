"""
Analysis and evaluation tools for research results.
Provides comprehensive analysis of experiment results for research purposes.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json
import subprocess
import os


class ResultsAnalyzer:
    """Comprehensive analysis of experiment results."""
    
    def __init__(self, experiments_dir: str = "experiments"):
        self.experiments_dir = Path(experiments_dir)
        self.analysis_dir = self.experiments_dir / "analysis"
        self.analysis_dir.mkdir(exist_ok=True)
    
    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        """Load all available experiment data."""
        data = {}
        
        # Load characteristics summary
        char_file = self.analysis_dir / "characteristics_summary.csv"
        if char_file.exists():
            data['characteristics'] = pd.read_csv(char_file)
            data['characteristics']['timestamp'] = pd.to_datetime(data['characteristics']['timestamp'])
        
        # Load OML summary
        oml_file = self.analysis_dir / "oml_summary.csv"
        if oml_file.exists():
            data['oml'] = pd.read_csv(oml_file)
            data['oml']['timestamp'] = pd.to_datetime(data['oml']['timestamp'])
        
        return data
    
    def generate_comprehensive_report(self) -> str:
        """Generate a comprehensive analysis report."""
        
        data = self.load_all_data()
        
        if not data:
            return "No experiment data found. Run some experiments first."
        
        report = []
        report.append("# Digital Twin Characteristics Extraction - Experiment Analysis Report")
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Characteristics Extraction Analysis
        if 'characteristics' in data:
            char_df = data['characteristics']
            report.append("## Characteristics Extraction Analysis")
            report.append(f"- **Total Experiments**: {len(char_df)}")
            report.append(f"- **Average Extraction Rate**: {char_df['extraction_rate'].mean():.2f}% ± {char_df['extraction_rate'].std():.2f}%")
            report.append(f"- **Best Extraction Rate**: {char_df['extraction_rate'].max():.2f}%")
            report.append(f"- **Average Processing Time**: {char_df['processing_time_seconds'].mean():.2f}s")
            report.append(f"- **Average Description Length**: {char_df['average_description_length'].mean():.0f} characters")
            report.append("")
            
            # Model performance comparison
            if len(char_df['model_name'].unique()) > 1:
                report.append("### Model Performance Comparison")
                model_stats = char_df.groupby('model_name').agg({
                    'extraction_rate': ['mean', 'std', 'count'],
                    'processing_time_seconds': ['mean', 'std'],
                    'average_description_length': 'mean'
                }).round(2)
                report.append(model_stats.to_string())
                report.append("")
            
            # Hyperparameter impact analysis
            report.append("### Hyperparameter Impact Analysis")
            
            # Chunk size analysis
            if len(char_df['chunk_size'].unique()) > 1:
                chunk_analysis = char_df.groupby('chunk_size')['extraction_rate'].agg(['mean', 'std', 'count'])
                report.append("#### Chunk Size Impact on Extraction Rate")
                report.append(chunk_analysis.to_string())
                report.append("")
            
            # # Retrieval K analysis
            #     report.append("#### Retrieval K Impact on Extraction Rate")
            #     report.append(k_analysis.to_string())
            #     report.append("")
            
            # Temperature analysis
            if len(char_df['temperature'].unique()) > 1:
                temp_analysis = char_df.groupby('temperature')['extraction_rate'].agg(['mean', 'std', 'count'])
                report.append("#### Temperature Impact on Extraction Rate")
                report.append(temp_analysis.to_string())
                report.append("")
        
        # OML Generation Analysis
        if 'oml' in data:
            oml_df = data['oml']
            report.append("## OML Generation Analysis")
            report.append(f"- **Total OML Generations**: {len(oml_df)}")
            report.append(f"- **Syntax Valid Rate**: {(oml_df['oml_syntax_valid'].sum() / len(oml_df) * 100):.1f}%")
            report.append(f"- **Average Completeness Score**: {oml_df['oml_completeness_score'].mean():.2f}")
            report.append(f"- **Average Instance Count**: {oml_df['oml_instance_count'].mean():.1f}")
            report.append(f"- **Average Generation Time**: {oml_df['generation_time_seconds'].mean():.2f}s")
            report.append("")
        
        # Quality trends over time
        if 'characteristics' in data and len(char_df) > 1:
            report.append("## Quality Trends Over Time")
            
            # Recent vs older experiments
            recent_threshold = datetime.now() - timedelta(days=7)
            recent_df = char_df[char_df['timestamp'] > recent_threshold]
            older_df = char_df[char_df['timestamp'] <= recent_threshold]
            
            if len(recent_df) > 0 and len(older_df) > 0:
                report.append(f"- **Recent (7 days) avg extraction rate**: {recent_df['extraction_rate'].mean():.2f}%")
                report.append(f"- **Older experiments avg extraction rate**: {older_df['extraction_rate'].mean():.2f}%")
                
                improvement = recent_df['extraction_rate'].mean() - older_df['extraction_rate'].mean()
                if improvement > 0:
                    report.append(f"- **Improvement**: +{improvement:.2f}% 📈")
                else:
                    report.append(f"- **Change**: {improvement:.2f}% 📉")
                report.append("")
        
        # Recommendations
        report.append("## Recommendations")
        
        if 'characteristics' in data:
            # Find best performing configuration
            best_exp = char_df.loc[char_df['extraction_rate'].idxmax()]
            
            report.append("### Best Performing Configuration")
            report.append(f"- **Model**: {best_exp['model_name']}")
            report.append(f"- **Chunk Size**: {best_exp['chunk_size']}")
            report.append(f"- **Chunk Overlap**: {best_exp['chunk_overlap']}")
            report.append(f"- **Temperature**: {best_exp['temperature']}")
            report.append(f"- **Extraction Rate**: {best_exp['extraction_rate']:.2f}%")
            report.append("")
            
            # Performance insights
            report.append("### Performance Insights")
            
            # Correlation analysis
            numeric_cols = ['chunk_size', 'chunk_overlap', 'temperature', 'extraction_rate']
            available_cols = [col for col in numeric_cols if col in char_df.columns]
            
            if len(available_cols) > 2:
                corr_with_extraction = char_df[available_cols].corr()['extraction_rate'].abs().sort_values(ascending=False)
                
                report.append("#### Factors most correlated with extraction rate:")
                for factor, correlation in corr_with_extraction.items():
                    if factor != 'extraction_rate':
                        report.append(f"- **{factor}**: {correlation:.3f}")
                report.append("")
        
        # Error analysis
        if 'characteristics' in data:
            error_rate = (char_df['error_count'] > 0).sum() / len(char_df) * 100
            report.append(f"### Error Analysis")
            report.append(f"- **Experiments with errors**: {error_rate:.1f}%")
            if error_rate > 0:
                report.append("- **Recommendation**: Review error logs for common issues")
            report.append("")
        
        return "\n".join(report)
    
    def create_dashboard_visualizations(self):
        """Create a comprehensive dashboard of visualizations."""
        
        data = self.load_all_data()
        
        if not data:
            print("No experiment data found. Run some experiments first.")
            return
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create comprehensive dashboard
        if 'characteristics' in data:
            char_df = data['characteristics']
            self._create_characteristics_dashboard(char_df)
        
        if 'oml' in data:
            oml_df = data['oml']
            self._create_oml_dashboard(oml_df)
        
        if 'characteristics' in data and 'oml' in data:
            self._create_comparison_dashboard(data['characteristics'], data['oml'])
    
    def _create_characteristics_dashboard(self, char_df: pd.DataFrame):
        """Create characteristics extraction dashboard."""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Characteristics Extraction Performance Dashboard', fontsize=16)
        
        # 1. Extraction rate over time
        axes[0, 0].plot(char_df['timestamp'], char_df['extraction_rate'], marker='o', alpha=0.7)
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Extraction Rate (%)')
        axes[0, 0].set_title('Extraction Rate Over Time')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Processing time vs extraction rate
        scatter = axes[0, 1].scatter(char_df['processing_time_seconds'], char_df['extraction_rate'], 
                                   c=char_df['chunk_size'], cmap='viridis', alpha=0.7)
        axes[0, 1].set_xlabel('Processing Time (seconds)')
        axes[0, 1].set_ylabel('Extraction Rate (%)')
        axes[0, 1].set_title('Processing Time vs Extraction Rate')
        plt.colorbar(scatter, ax=axes[0, 1], label='Chunk Size')
        
        # 3. Model comparison (if multiple models)
        if len(char_df['model_name'].unique()) > 1:
            model_stats = char_df.groupby('model_name')['extraction_rate'].mean().sort_values(ascending=True)
            axes[0, 2].barh(model_stats.index, model_stats.values)
            axes[0, 2].set_xlabel('Average Extraction Rate (%)')
            axes[0, 2].set_title('Model Performance Comparison')
        else:
            axes[0, 2].text(0.5, 0.5, 'Single Model Used', ha='center', va='center', transform=axes[0, 2].transAxes)
            axes[0, 2].set_title('Model Performance')
        
        # 4. Hyperparameter impact - Chunk size
        if len(char_df['chunk_size'].unique()) > 1:
            chunk_data = [char_df[char_df['chunk_size'] == cs]['extraction_rate'].values 
                         for cs in sorted(char_df['chunk_size'].unique())]
            axes[1, 0].boxplot(chunk_data, labels=sorted(char_df['chunk_size'].unique()))
            axes[1, 0].set_xlabel('Chunk Size')
            axes[1, 0].set_ylabel('Extraction Rate (%)')
            axes[1, 0].set_title('Chunk Size Impact')
        else:
            axes[1, 0].text(0.5, 0.5, 'Single Chunk Size', ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Chunk Size Impact')
        
        # 5. Temperature impact
        if len(char_df['temperature'].unique()) > 1:
            temp_stats = char_df.groupby('temperature')['extraction_rate'].mean()
            axes[1, 1].plot(temp_stats.index, temp_stats.values, marker='o', linewidth=2)
            axes[1, 1].set_xlabel('Temperature')
            axes[1, 1].set_ylabel('Average Extraction Rate (%)')
            axes[1, 1].set_title('Temperature Impact')
        else:
            axes[1, 1].text(0.5, 0.5, 'Single Temperature', ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Temperature Impact')
        
        # 6. Quality distribution
        axes[1, 2].hist(char_df['extraction_rate'], bins=20, alpha=0.7, edgecolor='black')
        axes[1, 2].axvline(char_df['extraction_rate'].mean(), color='red', linestyle='--', label='Mean')
        axes[1, 2].set_xlabel('Extraction Rate (%)')
        axes[1, 2].set_ylabel('Frequency')
        axes[1, 2].set_title('Extraction Rate Distribution')
        axes[1, 2].legend()
        
        plt.tight_layout()
        
        # Save dashboard
        dashboard_file = self.analysis_dir / "characteristics_dashboard.png"
        plt.savefig(dashboard_file, dpi=300, bbox_inches='tight')
        print(f"📊 Characteristics dashboard saved to: {dashboard_file}")
        
        plt.show()
    
    def _create_oml_dashboard(self, oml_df: pd.DataFrame):
        """Create OML generation dashboard."""
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('OML Generation Performance Dashboard', fontsize=16)
        
        # 1. Syntax validity over time
        axes[0, 0].plot(oml_df['timestamp'], oml_df['oml_syntax_valid'].astype(int), marker='o', alpha=0.7)
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Syntax Valid (1=Yes, 0=No)')
        axes[0, 0].set_title('OML Syntax Validity Over Time')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Completeness score distribution
        axes[0, 1].hist(oml_df['oml_completeness_score'], bins=15, alpha=0.7, edgecolor='black')
        axes[0, 1].axvline(oml_df['oml_completeness_score'].mean(), color='red', linestyle='--', label='Mean')
        axes[0, 1].set_xlabel('Completeness Score')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('OML Completeness Score Distribution')
        axes[0, 1].legend()
        
        # 3. Instance count vs completeness
        axes[1, 0].scatter(oml_df['oml_instance_count'], oml_df['oml_completeness_score'], alpha=0.7)
        axes[1, 0].set_xlabel('Instance Count')
        axes[1, 0].set_ylabel('Completeness Score')
        axes[1, 0].set_title('Instance Count vs Completeness')
        
        # 4. Generation time distribution
        axes[1, 1].hist(oml_df['generation_time_seconds'], bins=15, alpha=0.7, edgecolor='black')
        axes[1, 1].axvline(oml_df['generation_time_seconds'].mean(), color='red', linestyle='--', label='Mean')
        axes[1, 1].set_xlabel('Generation Time (seconds)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('OML Generation Time Distribution')
        axes[1, 1].legend()
        
        plt.tight_layout()
        
        # Save dashboard
        dashboard_file = self.analysis_dir / "oml_dashboard.png"
        plt.savefig(dashboard_file, dpi=300, bbox_inches='tight')
        print(f"📊 OML dashboard saved to: {dashboard_file}")
        
        plt.show()
    
    def _create_comparison_dashboard(self, char_df: pd.DataFrame, oml_df: pd.DataFrame):
        """Create comparison dashboard between characteristics and OML tasks."""
        
        # Merge data on experiment ID if possible
        if 'characteristics_experiment_id' in oml_df.columns:
            merged_df = char_df.merge(
                oml_df, 
                left_on='experiment_id', 
                right_on='characteristics_experiment_id', 
                how='inner',
                suffixes=('_char', '_oml')
            )
            
            if len(merged_df) > 0:
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                fig.suptitle('Characteristics vs OML Performance Comparison', fontsize=16)
                
                # 1. Extraction rate vs OML completeness
                axes[0].scatter(merged_df['extraction_rate'], merged_df['oml_completeness_score'], alpha=0.7)
                axes[0].set_xlabel('Characteristics Extraction Rate (%)')
                axes[0].set_ylabel('OML Completeness Score')
                axes[0].set_title('Extraction Quality vs OML Quality')
                
                # Add correlation coefficient
                corr = merged_df['extraction_rate'].corr(merged_df['oml_completeness_score'])
                axes[0].text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=axes[0].transAxes, 
                           bbox=dict(boxstyle="round", facecolor='white', alpha=0.8))
                
                # 2. Processing time comparison
                time_comparison = merged_df[['processing_time_seconds', 'generation_time_seconds']].mean()
                axes[1].bar(time_comparison.index, time_comparison.values)
                axes[1].set_ylabel('Average Time (seconds)')
                axes[1].set_title('Processing Time Comparison')
                axes[1].set_xticklabels(['Characteristics\\nExtraction', 'OML\\nGeneration'], rotation=45)
                
                plt.tight_layout()
                
                # Save comparison dashboard
                comparison_file = self.analysis_dir / "comparison_dashboard.png"
                plt.savefig(comparison_file, dpi=300, bbox_inches='tight')
                print(f"📊 Comparison dashboard saved to: {comparison_file}")
                
                plt.show()
    
    def export_research_summary(self) -> str:
        """Export a research-ready summary of all experiments."""
        
        # Generate comprehensive report
        report = self.generate_comprehensive_report()
        
        # Save report with proper newlines
        report_file = self.analysis_dir / f"research_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"📄 Research summary saved to: {report_file}")
        
        return str(report_file)
    
    def validate_with_oml_tools(self, oml_tools_path: Path, catalog_path: Path, 
                            output_path: Path) -> Dict[str, Any]:
        """Use oml-tools directly for validation."""
        
        try:
            # Run the validation
            result = subprocess.run([
                str(oml_tools_path / "gradlew.bat" if os.name == 'nt' else "gradlew"),
                "oml-validate:run",
                f"--args=-i {catalog_path} -o {output_path}"
            ], 
            cwd=oml_tools_path,
            capture_output=True, 
            text=True, 
            timeout=300)
            
            # Read the report
            report = ""
            if output_path.exists():
                with open(output_path, 'r') as f:
                    report = f.read()
            
            return {
                'success': result.returncode == 0,
                'return_code': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'report': report
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _analyze_oml_quality(self, oml_content: str) -> Dict[str, Any]:
        """Updated method using OpenCAESAR validation."""
        
        # Use the new validator
        validation_result = self.validate_with_oml_tools(
            oml_tools_path=Path("../oml-tools"),
            catalog_path=Path("../data/DTOnto/catalog.xml").resolve(),
            output_path=Path("validation_report.txt")
        )
        
        # Convert to expected format for backward compatibility
        return {
            'syntax_valid': validation_result.get('success', False),
            'completeness_score': 1.0 if validation_result.get('success') else 0.0,
            'line_count': oml_content.count('\n') + 1,
            'instance_count': oml_content.count('instance ')
        }


def main():
    """Main function for results analysis."""
    
    print("📊 Digital Twin Experiments - Results Analysis")
    print("=" * 60)
    
    # Create analyzer
    analyzer = ResultsAnalyzer()
    
    # Generate comprehensive report
    print("📄 Generating comprehensive analysis report...")
    report_file = analyzer.export_research_summary()
    
    # Create visualizations
    print("📊 Creating dashboard visualizations...")
    analyzer.create_dashboard_visualizations()
    
    print(f"\n✅ Analysis completed!")
    print(f"📄 Report: {report_file}")
    print(f"📊 Dashboards: experiments/analysis/")
    print("\n💡 Use these results to:")
    print("   • Compare different hyperparameter configurations")
    print("   • Track performance improvements over time")
    print("   • Identify optimal settings for your use case")
    print("   • Prepare research publications with quantitative results")


if __name__ == "__main__":
    main()
