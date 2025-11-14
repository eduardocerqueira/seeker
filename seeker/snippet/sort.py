#date: 2025-11-14T16:42:55Z
#url: https://api.github.com/gists/9f2238a71eced03823a01dc41b9c3383
#owner: https://api.github.com/users/xeonrag

from dotsermodz import app
from pyrogram import Client, filters
from pyrogram.types import Message
from config import SUDO
import re
import os
from datetime import datetime

def extract_numbers_from_text(text):
    """Extract phone numbers (8-15 digits) from text"""
    numbers = []
    lines = text.split('\n')
    
    for line in lines:
        clean_line = line.strip()
        if clean_line:
            number_match = re.search(r'\d{8,15}', clean_line)
            if number_match:
                numbers.append(number_match.group(0))
    
    return numbers

def find_fancy_patterns(number):
    """Find fancy patterns in a number"""
    patterns = []
    num_str = str(number)
    
    # Consecutive repeats (e.g., 111, 2222)
    consecutive_repeats = re.findall(r'(\d)\1{2,}', num_str)
    for match in consecutive_repeats:
        full_match = re.search(f'({re.escape(match)})+', num_str)
        if full_match:
            pattern_str = full_match.group(0)
            patterns.append({
                'type': 'Consecutive Repeats',
                'pattern': pattern_str,
                'score': len(pattern_str) * 10
            })
    
    # Sequential up (e.g., 0123, 1234)
    sequential_up = re.findall(r'(?:0123|1234|2345|3456|4567|5678|6789)', num_str)
    for match in sequential_up:
        patterns.append({
            'type': 'Sequential Up',
            'pattern': match,
            'score': len(match) * 8
        })
    
    # Sequential down (e.g., 9876, 8765)
    sequential_down = re.findall(r'(?:9876|8765|7654|6543|5432|4321|3210)', num_str)
    for match in sequential_down:
        patterns.append({
            'type': 'Sequential Down',
            'pattern': match,
            'score': len(match) * 8
        })
    
    # Double digit patterns (e.g., 121212)
    double_digits = re.findall(r'(\d\d)\1+', num_str)
    for match in double_digits:
        full_match = re.search(f'({re.escape(match)})+', num_str)
        if full_match:
            pattern_str = full_match.group(0)
            patterns.append({
                'type': 'Double Pattern',
                'pattern': pattern_str,
                'score': len(pattern_str) * 6
            })
    
    # Palindrome (e.g., 1221, 3443)
    palindromes = re.findall(r'(\d)(\d)\2\1', num_str)
    for match in palindromes:
        pattern_str = ''.join(match)
        patterns.append({
            'type': 'Palindrome',
            'pattern': pattern_str,
            'score': 15
        })
    
    # Alternating (e.g., 1212, 3434)
    alternating = re.findall(r'(\d)(\d)\1\2', num_str)
    for match in alternating:
        pattern_str = ''.join(match)
        patterns.append({
            'type': 'Alternating',
            'pattern': pattern_str,
            'score': 12
        })
    
    return patterns

def analyze_fancy_numbers(numbers):
    """Analyze all numbers and find fancy ones"""
    fancy_numbers = []
    
    for number in numbers:
        patterns = find_fancy_patterns(number)
        
        if patterns:
            total_score = sum(p['score'] for p in patterns)
            
            fancy_numbers.append({
                'number': number,
                'patterns': patterns,
                'score': total_score,
                'pattern_count': len(patterns)
            })
    
    # Sort by score (highest first)
    return sorted(fancy_numbers, key=lambda x: x['score'], reverse=True)

@app.on_message(filters.command("sort") & filters.user(SUDO))
async def sort_fancy_numbers(client: Client, message: Message):
    """Find fancy numbers from a text file"""
    try:
        # Check if replying to a document
        if not message.reply_to_message or not message.reply_to_message.document:
            await message.reply(
                "📄 Reply to a text file to find fancy numbers\n\n"
                "*Usage:* Reply to .txt file with /sort"
            )
            return
        
        # Check if it's a text file
        doc = message.reply_to_message.document
        if doc.mime_type != 'text/plain':
            await message.reply("❌ Please reply to a .txt file only")
            return
        
        # Download the file
        status_msg = await message.reply("🔍 Processing file and searching for fancy numbers...")
        file_path = await message.reply_to_message.download()
        
        # Read file content
        with open(file_path, 'r', encoding='utf-8') as f:
            file_content = f.read()
        
        # Extract numbers
        numbers = extract_numbers_from_text(file_content)
        
        if not numbers:
            os.remove(file_path)
            await status_msg.edit("❌ No valid numbers found in the file")
            return
        
        # Analyze fancy numbers
        fancy_numbers = analyze_fancy_numbers(numbers)
        
        if not fancy_numbers:
            os.remove(file_path)
            await status_msg.edit(
                f"📊 *Analysis Complete*\n\n"
                f"🔍 *Total Numbers:* {len(numbers)}\n"
                f"❌ *Fancy Numbers:* 0\n\n"
                f"_No fancy patterns found_"
            )
            return
        
        # Create result text
        result_text = "✨ *Fancy Numbers Found*\n\n"
        result_text += f"🔍 *Total Checked:* {len(numbers)}\n"
        result_text += f"✨ *Fancy Found:* {len(fancy_numbers)}\n\n"
        result_text += "🏆 *Top 50 Fancy Numbers:*\n"
        result_text += "```\n"
        
        top_50 = fancy_numbers[:50]
        for item in top_50:
            result_text += f"{item['number']}\n"
        
        result_text += "```"
        
        if len(fancy_numbers) > 50:
            result_text += f"\n_And {len(fancy_numbers) - 50} more fancy numbers in file..._"
        
        # If too many results, send as file
        if len(fancy_numbers) > 50 or len(result_text) > 4000:
            file_name = f"fancy_numbers_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            output_path = f"./{file_name}"
            
            # Create output file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("Fancy Numbers (Sorted by Score)\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Total Numbers Analyzed: {len(numbers)}\n")
                f.write(f"Fancy Numbers Found: {len(fancy_numbers)}\n\n")
                
                for item in fancy_numbers:
                    f.write(f"{item['number']}\n")
            
            # Send file
            await message.reply_document(
                document=output_path,
                caption=(
                    f"✨ *Complete Fancy Numbers List*\n\n"
                    f"🔍 *Total Analyzed:* {len(numbers)}\n"
                    f"✨ *Fancy Found:* {len(fancy_numbers)}"
                )
            )
            
            # Clean up
            os.remove(output_path)
        
        # Send result message
        await status_msg.edit(result_text)
        
        # Clean up downloaded file
        os.remove(file_path)
        
    except Exception as error:
        print(f"Fancy Number Error: {error}")
        await message.reply("💥 An error occurred while processing the file")
        # Clean up in case of error
        if 'file_path' in locals() and os.path.exists(file_path):
            os.remove(file_path)