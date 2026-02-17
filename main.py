"""
Main entry point for AI Financial Report Generator.
Usage: python main.py --file <file_path> --output <output_path> [options]
"""

import argparse
import logging
from pathlib import Path
from datetime import datetime
from src.agents.coordinator_agent import CoordinatorAgent
from src.config import CONFIG, LLMBackend
from src.core.state import MEMORY

# Setup logging
logging.basicConfig(
    level=logging.INFO if not CONFIG.debug_mode else logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def main():
    """Main execution entry point."""
    parser = argparse.ArgumentParser(
        description='AI Financial Report Generator - Multi-Agent System'
    )

    # File input
    parser.add_argument(
        '--file', '-f',
        type=str,
        required=True,
        help='Path to financial data file (CSV, PDF, DOCX, XLSX, TXT)'
    )

    # Output options
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='outputs/report.pdf',
        help='Output PDF file path (default: outputs/report.pdf)'
    )

    # Report options
    parser.add_argument(
        '--summary',
        action='store_true',
        default=True,
        help='Include executive summary'
    )

    parser.add_argument(
        '--charts',
        action='store_true',
        default=True,
        help='Include charts and visualizations'
    )

    parser.add_argument(
        '--kpis',
        action='store_true',
        default=True,
        help='Include KPI metrics'
    )

    parser.add_argument(
        '--chart-types',
        type=str,
        default='line,bar,pie',
        help='Comma-separated chart types (line, bar, pie)'
    )

    # LLM configuration
    parser.add_argument(
        '--llm-backend',
        type=str,
        default='ollama',
        choices=['openai', 'gemini', 'ollama', 'lm_studio', 'anthropic'],
        help='LLM backend to use'
    )

    parser.add_argument(
        '--llm-model',
        type=str,
        default='mistral:latest',
        help='LLM model name'
    )

    parser.add_argument(
        '--llm-api-key',
        type=str,
        help='API key for LLM backend (if required)'
    )

    # Branding
    parser.add_argument(
        '--company-name',
        type=str,
        default='Financial Analytics Corp',
        help='Company name for report branding'
    )

    parser.add_argument(
        '--primary-color',
        type=str,
        default='#1f77b4',
        help='Primary color (hex format)'
    )

    # System options
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode'
    )

    parser.add_argument(
        '--skip-tests',
        action='store_true',
        help='Skip automated testing phase'
    )

    args = parser.parse_args()

    # Validate input file
    input_file = Path(args.file)
    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        return False

    # Configure system
    try:
        CONFIG.llm.backend = LLMBackend[args.llm_backend.upper()]
        CONFIG.llm.model_name = args.llm_model
        if args.llm_api_key:
            CONFIG.llm.api_key = args.llm_api_key
        CONFIG.branding.company_name = args.company_name
        CONFIG.branding.primary_color = args.primary_color
        CONFIG.debug_mode = args.debug
    except Exception as e:
        logger.error(f"Configuration error: {str(e)}")
        return False

    # Prepare output directory
    Path("outputs").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    # Create execution task
    task = {
        "task_id": f"report_{datetime.now().timestamp()}",
        "file_path": str(input_file),
        "output_path": args.output,
        "report_type": "financial",
        "include_summary": args.summary,
        "include_charts": args.charts,
        "include_kpis": args.kpis,
        "chart_types": args.chart_types.split(","),
        "skip_tests": args.skip_tests
    }

    logger.info(f"Starting report generation: {task['task_id']}")
    logger.info(f"Input file: {input_file}")
    logger.info(f"LLM Backend: {CONFIG.llm.backend.value} ({CONFIG.llm.model_name})")

    # Execute coordinator
    try:
        coordinator = CoordinatorAgent()
        logger.info("Coordinator initialized")

        result = coordinator.execute(task)

        if result.success:
            logger.info("✅ Report generation completed successfully")
            logger.info(f"Status: {result.data.get('status')}")
            logger.info(f"Output: {args.output}")

            # Print summary
            print("\n" + "="*60)
            print("REPORT GENERATION COMPLETE")
            print("="*60)
            print(f"Task ID: {task['task_id']}")
            print(f"Status: {result.data.get('status')}")
            print(f"Duration: {result.duration_seconds:.2f} seconds")
            print(f"Output: {args.output}")
            print(f"Test Status: {result.data.get('test_status')}")
            print("="*60 + "\n")

            return True
        else:
            logger.error(f"❌ Report generation failed: {result.error}")
            return False

    except Exception as e:
        logger.error(f"Fatal error: {str(e)}", exc_info=True)
        return False

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
