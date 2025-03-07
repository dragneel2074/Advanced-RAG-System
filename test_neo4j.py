import os
from knowledge_graph import KnowledgeGraph
from dotenv import load_dotenv
import uuid

# Load environment variables
load_dotenv()

def test_neo4j_connection():
    """Test the Neo4j connection and basic functionality."""
    print("Testing Neo4j connection...")
    
    # Force initialization
    kg = KnowledgeGraph(force_init=True)
    
    if not kg.connected:
        print("❌ Failed to connect to Neo4j")
        return False
    
    print("✅ Connected to Neo4j successfully")
    
    # Test classification
    test_message = "I'm having trouble with the login button on your website"
    sub_issue, main_issue = kg.classify_message(test_message)
    print(f"✅ Classification test: '{test_message}' → {main_issue} → {sub_issue}")
    
    # Test adding a question
    question_id = str(uuid.uuid4())
    success = kg.add_question(
        question_id=question_id,
        content="How do I reset my password?",
        chat_id="test-chat-id"
    )
    
    if success:
        print(f"✅ Added test question with ID: {question_id}")
    else:
        print("❌ Failed to add test question")
        return False
    
    # Test getting similar questions
    similar = kg.get_similar_questions("Login Problem", limit=5)
    print(f"✅ Retrieved {len(similar)} similar questions")
    
    # Test statistics
    stats = kg.get_category_statistics()
    print("✅ Retrieved category statistics:")
    for issue, data in stats.items():
        print(f"  - {issue}: {data['total']} questions")
        for subissue, count in data['subissues'].items():
            print(f"    - {subissue}: {count}")
    
    # Test relationship existence
    relationships_exist = kg._check_relationships_exist()
    print(f"✅ Required relationships exist: {relationships_exist}")
    
    # Close connection
    kg.close()
    print("✅ Test completed successfully")
    return True

if __name__ == "__main__":
    test_neo4j_connection() 