from agent import core_agent, ACTIVE_CONFIG, ACTIVE_CONFIG_NAME
from tools import (
    save_markdown_curriculum,
    save_research_metadata,
    print_research_summary,
    create_quick_reference
)
from google.adk.sessions import Session, InMemorySessionService
from google.adk.runners import Runner
from google.genai import types
from google.adk.agents import Agent
import asyncio
import sys
import os
import logging
from datetime import datetime
from typing import Optional

from config import print_config

# Add the current directory to sys.path to ensure we can import agent.py
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Configure logging
log_file_path = ACTIVE_CONFIG.log_file
log_dir = os.path.dirname(log_file_path)
if log_dir and not os.path.exists(log_dir):
    os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def get_user_input() -> str:
    """
    Interactively get research topic from user with validation.

    Returns:
        The validated research topic
    """
    print("\n" + "="*80)
    print("🔬 CEDLM AUTONOMOUS RESEARCHER")
    print("="*80)
    print("\nWelcome to the Autonomous Deep Research Agent!")
    print("This agent will conduct rigorous, iterative research on any topic")
    print("and generate a comprehensive markdown curriculum.\n")

    while True:
        print("Enter your research topic (or 'quit' to exit):")
        topic = input("📋 Topic: ").strip()

        if topic.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye!")
            sys.exit(0)

        if not topic:
            print("⚠️  Topic cannot be empty. Please try again.\n")
            continue

        if len(topic) < 3:
            print("⚠️  Topic too short. Please provide more detail.\n")
            continue

        # Confirm with user
        print(f"\n✅ Research topic: '{topic}'")
        confirm = input("Proceed with this topic? (y/n): ").strip().lower()

        if confirm in ['y', 'yes']:
            return topic
        else:
            print("\nLet's try again.\n")


def parse_command_line_args() -> Optional[str]:
    """
    Parse command line arguments for research topic.

    Returns:
        The research topic or None if not provided
    """
    if len(sys.argv) > 1:
        # Join all arguments as the topic
        topic = " ".join(sys.argv[1:])

        # Skip if it's a flag
        if topic.startswith('-'):
            return None

        return topic

    return None


async def run_agent(
    agent: Agent,
    query: str,
    session: Session,
    user_id: str,
    session_service: InMemorySessionService,
):
    """
    Run the autonomous research agent with full lifecycle management.

    Args:
        agent: The agent to run
        query: The research query/topic
        session: The session object
        user_id: User identifier
        session_service: Session service for state management

    Returns:
        The agent's response
    """
    logger.info(f"Starting autonomous research for topic: {query}")
    logger.info(f"User ID: {user_id}")
    logger.info("Active research preset: %s", ACTIVE_CONFIG_NAME)

    # Store the topic in session state for tracking
    session.state["research_topic"] = query

    runner = Runner(
        agent=agent,
        session_service=session_service,
        app_name=agent.name,
    )

    print("\n" + "="*80)
    print("🚀 CEDLM AUTONOMOUS RESEARCHER")
    print("="*80)
    print(f"📋 Research Topic: {query}")
    print(f"⏰ Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"👤 User: {user_id}")
    print("="*80 + "\n")

    print("🔄 Initiating autonomous research process...")
    print("   The agent will iteratively research until deep understanding is achieved.")
    print("   This may take several minutes depending on topic complexity.\n")

    try:
        # Run the agent (the IterativeResearchAgent handles all iterations internally)
        response = runner.run(
            user_id=user_id, new_message=query, session_id=session)

        print("\n✅ Research process completed!\n")

        # Extract state from session
        state = dict(session.state)

        # Print research summary
        print_research_summary(state)

        # Save outputs
        output_dir = os.path.join(current_dir, ACTIVE_CONFIG.output_dir)

        # Save the markdown curriculum
        markdown_content = state.get("final_markdown_curriculum", "")
        if markdown_content:
            curriculum_path = save_markdown_curriculum(
                markdown_content=markdown_content,
                topic=query,
                output_dir=output_dir
            )
            print(f"\n📄 Main curriculum: {curriculum_path}")
        else:
            logger.warning("No markdown curriculum generated")
            print("\n⚠️  Warning: No markdown curriculum was generated")

        # Save research metadata
        metadata_path = save_research_metadata(
            state=state,
            output_dir=output_dir,
            topic=query
        )
        print(f"📊 Metadata: {metadata_path}")

        # Create quick reference
        quick_ref_path = create_quick_reference(
            state=state,
            output_dir=output_dir
        )
        print(f"⚡ Quick reference: {quick_ref_path}")

        print("\n" + "="*80)
        print("✨ All outputs saved successfully!")
        print("="*80 + "\n")

        return response

    except Exception as e:
        logger.error(f"Error during research process: {e}", exc_info=True)
        print(f"\n❌ Error occurred: {e}")
        print("Check cedlm_research.log for details.")
        raise


async def main():
    """
    Main entry point for the autonomous researcher.
    """
    print("\n" + "="*80)
    print("🎓 CEDLM AUTONOMOUS DEEP RESEARCH AGENT")
    print("="*80)
    print("Powered by Google Agent Development Kit & Gemini 3 Pro\n")
    print("="*80 + "\n")
    print(f"🛠️ Active research preset: {ACTIVE_CONFIG_NAME}")
    print_config(ACTIVE_CONFIG)

    # Get research topic from command line or interactive input
    topic = parse_command_line_args()

    if topic:
        print(f"📋 Topic from command line: {topic}\n")
    else:
        # Show usage examples
        print("💡 Usage Examples:")
        print("   python main.py 'Machine Learning Transformers'")
        print("   python main.py 'Quantum Computing Fundamentals'")
        print("   python main.py 'Advanced Rust Memory Management'\n")

        # Get interactive input
        topic = get_user_input()

    # Initialize session service
    session_service = InMemorySessionService()
    user_id = f"researcher_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    session = await session_service.create_session(
        app_name=core_agent.name, user_id=user_id)

    # Run the autonomous research agent
    try:
        await run_agent(
            agent=core_agent,
            query=topic,
            session=session,
            user_id=user_id,
            session_service=session_service
        )

        logger.info("Research completed successfully")

        # Offer to research another topic
        print("\n" + "="*80)
        print("🎉 Research session complete!")
        print("="*80)
        print("\nWould you like to research another topic?")
        another = input("Research another topic? (y/n): ").strip().lower()

        if another in ['y', 'yes']:
            print("\n🔄 Starting new research session...\n")
            await main()  # Recursive call for new session

    except KeyboardInterrupt:
        print("\n\n⚠️  Research interrupted by user")
        logger.info("Research interrupted by user")
        print("\nPartial results may have been saved to the output directory.")
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        logger.error(f"Fatal error in main: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    # Display banner
    print("\n")
    print("╔═══════════════════════════════════════════════════════════════════════════════╗")
    print("║                                                                               ║")
    print("║                    CEDLM AUTONOMOUS RESEARCHER v2.0                           ║")
    print("║                                                                               ║")
    print("║           Rigorous Iterative Research Until Deep Understanding                ║")
    print("║                                                                               ║")
    print("╚═══════════════════════════════════════════════════════════════════════════════╝")

    # Run the async main function
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n\n❌ Critical error: {e}")
        sys.exit(1)
