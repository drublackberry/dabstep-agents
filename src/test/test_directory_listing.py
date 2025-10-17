#!/usr/bin/env python3
"""
Test the get_directory_listing function to verify it correctly reads directory contents.
"""

import json
from pathlib import Path
from agents.multi_agent_system import get_directory_listing


def test_directory_listing():
    """Test that get_directory_listing correctly reads and categorizes files."""
    
    # Test with the data/context directory
    test_path = str(Path(__file__).parent.parent.parent / "data" / "context")
    
    print("="*80)
    print("Testing get_directory_listing()")
    print("="*80)
    
    print(f"\nTest directory: {test_path}")
    
    # Get directory listing
    listing = get_directory_listing(test_path)
    
    print(f"\n📊 Directory Listing Results:")
    print(f"   Total files: {listing.get('total_files', 0)}")
    print(f"   Total directories: {listing.get('total_directories', 0)}")
    
    if listing.get('error'):
        print(f"\n❌ Error: {listing['error']}")
        return
    
    print(f"\n📄 Files found:")
    for file_info in listing.get('files', []):
        print(f"\n   Name: {file_info['name']}")
        print(f"   Path: {file_info['path']}")
        print(f"   Size: {file_info['size_bytes']} bytes")
        print(f"   Extension: {file_info['extension']}")
        print(f"   Category: {file_info['category']}")
    
    if listing.get('directories'):
        print(f"\n📁 Directories found:")
        for dir_info in listing.get('directories', []):
            print(f"   - {dir_info['name']}")
    
    print("\n" + "="*80)
    print("Full JSON output:")
    print("="*80)
    print(json.dumps(listing, indent=2))
    
    print("\n✅ Test completed!")
    
    # Verify expected structure
    assert 'directory_path' in listing, "Missing directory_path"
    assert 'total_files' in listing, "Missing total_files"
    assert 'files' in listing, "Missing files list"
    assert isinstance(listing['files'], list), "files should be a list"
    
    # Verify file info structure
    for file_info in listing['files']:
        assert 'name' in file_info, "File missing name"
        assert 'path' in file_info, "File missing path"
        assert 'size_bytes' in file_info, "File missing size_bytes"
        assert 'extension' in file_info, "File missing extension"
        assert 'category' in file_info, "File missing category"
        assert file_info['category'] in ['data', 'documentation', 'code', 'other'], \
            f"Invalid category: {file_info['category']}"
    
    print("\n✅ All assertions passed!")


def test_nonexistent_directory():
    """Test handling of non-existent directory."""
    
    print("\n" + "="*80)
    print("Testing with non-existent directory")
    print("="*80)
    
    listing = get_directory_listing("/nonexistent/path/to/directory")
    
    print(f"\n📊 Result:")
    print(json.dumps(listing, indent=2))
    
    assert listing['total_files'] == 0, "Should have 0 files"
    assert 'status' in listing, "Should have status field"
    assert 'error' in listing['status'].lower(), "Status should contain 'error' keyword"
    
    print("\n✅ Non-existent directory handled correctly!")


if __name__ == "__main__":
    test_directory_listing()
    test_nonexistent_directory()
    
    print("\n" + "="*80)
    print("All tests passed! ✅")
    print("="*80)
