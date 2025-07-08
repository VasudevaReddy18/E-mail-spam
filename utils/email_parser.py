import email
from email import policy
from email.parser import BytesParser
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import re
import base64
import quopri
from typing import Dict, List, Optional, Tuple
import os

class EmailParser:
    """
    Advanced email parser for spam classification
    """
    
    def __init__(self):
        """Initialize the email parser"""
        self.parser = BytesParser(policy=policy.default)
        
        # Common email headers to extract
        self.important_headers = {
            'from', 'to', 'subject', 'date', 'message-id',
            'reply-to', 'return-path', 'x-mailer', 'user-agent',
            'content-type', 'content-transfer-encoding',
            'x-priority', 'x-msmail-priority', 'importance'
        }
        
        # Spam-related headers
        self.spam_headers = {
            'x-spam-status', 'x-spam-score', 'x-spam-flag',
            'x-spam-checker-version', 'x-spam-report',
            'x-virus-scanned', 'x-virus-status'
        }
    
    def parse_email_file(self, file_path: str) -> Dict:
        """
        Parse email from file
        """
        try:
            with open(file_path, 'rb') as f:
                email_data = f.read()
            return self.parse_email_bytes(email_data)
        except Exception as e:
            return {'error': f'Failed to parse email file: {str(e)}'}
    
    def parse_email_bytes(self, email_bytes: bytes) -> Dict:
        """
        Parse email from bytes
        """
        try:
            msg = self.parser.parsebytes(email_bytes)
            return self._extract_email_data(msg)
        except Exception as e:
            return {'error': f'Failed to parse email bytes: {str(e)}'}
    
    def parse_email_string(self, email_string: str) -> Dict:
        """
        Parse email from string
        """
        try:
            msg = email.message_from_string(email_string)
            return self._extract_email_data(msg)
        except Exception as e:
            return {'error': f'Failed to parse email string: {str(e)}'}
    
    def _extract_email_data(self, msg) -> Dict:
        """
        Extract comprehensive data from email message
        """
        email_data = {
            'headers': {},
            'body': {
                'text': '',
                'html': '',
                'plain_text': ''
            },
            'attachments': [],
            'metadata': {},
            'spam_indicators': {}
        }
        
        # Extract headers
        email_data['headers'] = self._extract_headers(msg)
        
        # Extract body content
        email_data['body'] = self._extract_body(msg)
        
        # Extract attachments
        email_data['attachments'] = self._extract_attachments(msg)
        
        # Extract metadata
        email_data['metadata'] = self._extract_metadata(msg)
        
        # Extract spam indicators
        email_data['spam_indicators'] = self._extract_spam_indicators(msg)
        
        return email_data
    
    def _extract_headers(self, msg) -> Dict:
        """Extract and process email headers"""
        headers = {}
        
        for header_name, header_value in msg.items():
            header_name_lower = header_name.lower()
            
            # Store important headers
            if header_name_lower in self.important_headers:
                headers[header_name_lower] = header_value
            
            # Store spam-related headers
            if header_name_lower in self.spam_headers:
                headers[header_name_lower] = header_value
        
        return headers
    
    def _extract_body(self, msg) -> Dict:
        """Extract email body content"""
        body_data = {
            'text': '',
            'html': '',
            'plain_text': ''
        }
        
        if msg.is_multipart():
            # Handle multipart messages
            for part in msg.walk():
                content_type = part.get_content_type()
                content_disposition = str(part.get('Content-Disposition', ''))
                
                # Skip attachments
                if 'attachment' in content_disposition:
                    continue
                
                # Extract text content
                if content_type == 'text/plain':
                    body_data['plain_text'] += self._decode_content(part)
                elif content_type == 'text/html':
                    body_data['html'] += self._decode_content(part)
        else:
            # Handle single part messages
            content_type = msg.get_content_type()
            if content_type == 'text/plain':
                body_data['plain_text'] = self._decode_content(msg)
            elif content_type == 'text/html':
                body_data['html'] = self._decode_content(msg)
            else:
                body_data['text'] = self._decode_content(msg)
        
        # Combine all text content
        body_data['text'] = body_data['plain_text'] + ' ' + body_data['html']
        
        return body_data
    
    def _decode_content(self, part) -> str:
        """Decode email content based on encoding"""
        try:
            payload = part.get_payload(decode=True)
            if payload is None:
                return ""
            
            # Get charset
            charset = part.get_content_charset() or 'utf-8'
            
            # Handle different encodings
            if charset.lower() in ['utf-8', 'ascii']:
                return payload.decode(charset, errors='ignore')
            else:
                # Try to decode with different encodings
                for encoding in ['utf-8', 'latin-1', 'cp1252', 'ascii']:
                    try:
                        return payload.decode(encoding, errors='ignore')
                    except UnicodeDecodeError:
                        continue
                
                # Fallback to latin-1
                return payload.decode('latin-1', errors='ignore')
        
        except Exception as e:
            return f"Error decoding content: {str(e)}"
    
    def _extract_attachments(self, msg) -> List[Dict]:
        """Extract attachment information"""
        attachments = []
        
        if msg.is_multipart():
            for part in msg.walk():
                content_disposition = str(part.get('Content-Disposition', ''))
                
                if 'attachment' in content_disposition:
                    attachment = {
                        'filename': part.get_filename(),
                        'content_type': part.get_content_type(),
                        'size': len(part.get_payload(decode=True) or b''),
                        'content_disposition': content_disposition
                    }
                    attachments.append(attachment)
        
        return attachments
    
    def _extract_metadata(self, msg) -> Dict:
        """Extract email metadata"""
        metadata = {
            'is_multipart': msg.is_multipart(),
            'content_type': msg.get_content_type(),
            'content_maintype': msg.get_content_maintype(),
            'content_subtype': msg.get_content_subtype(),
            'has_attachments': False,
            'attachment_count': 0,
            'total_size': 0
        }
        
        # Check for attachments
        if msg.is_multipart():
            for part in msg.walk():
                content_disposition = str(part.get('Content-Disposition', ''))
                if 'attachment' in content_disposition:
                    metadata['has_attachments'] = True
                    metadata['attachment_count'] += 1
                    metadata['total_size'] += len(part.get_payload(decode=True) or b'')
        
        return metadata
    
    def _extract_spam_indicators(self, msg) -> Dict:
        """Extract spam-related indicators from email"""
        indicators = {
            'spam_score': 0,
            'spam_status': '',
            'virus_scanned': False,
            'virus_status': '',
            'priority': 'normal',
            'importance': 'normal'
        }
        
        # Extract spam headers
        spam_status = msg.get('X-Spam-Status', '')
        if spam_status:
            indicators['spam_status'] = spam_status
            # Extract score from spam status
            score_match = re.search(r'score=([\d.-]+)', spam_status)
            if score_match:
                indicators['spam_score'] = float(score_match.group(1))
        
        # Extract virus scanning info
        virus_scanned = msg.get('X-Virus-Scanned', '')
        if virus_scanned:
            indicators['virus_scanned'] = True
            indicators['virus_status'] = msg.get('X-Virus-Status', '')
        
        # Extract priority information
        priority = msg.get('X-Priority', '')
        if priority:
            indicators['priority'] = priority
        
        importance = msg.get('Importance', '')
        if importance:
            indicators['importance'] = importance
        
        return indicators
    
    def get_email_content_for_classification(self, email_data: Dict) -> str:
        """
        Extract the best content for spam classification
        """
        # Priority: plain text > HTML > combined text
        if email_data['body']['plain_text']:
            return email_data['body']['plain_text']
        elif email_data['body']['html']:
            return email_data['body']['html']
        else:
            return email_data['body']['text']
    
    def extract_sender_domain(self, email_data: Dict) -> str:
        """Extract sender domain from email"""
        from_header = email_data['headers'].get('from', '')
        if from_header:
            # Extract email from "Name <email@domain.com>" format
            email_match = re.search(r'<(.+?)>', from_header)
            if email_match:
                email_address = email_match.group(1)
            else:
                email_address = from_header
            
            # Extract domain
            domain_match = re.search(r'@(.+)$', email_address)
            if domain_match:
                return domain_match.group(1)
        
        return ''
    
    def extract_recipient_domain(self, email_data: Dict) -> str:
        """Extract recipient domain from email"""
        to_header = email_data['headers'].get('to', '')
        if to_header:
            # Extract email from "Name <email@domain.com>" format
            email_match = re.search(r'<(.+?)>', to_header)
            if email_match:
                email_address = email_match.group(1)
            else:
                email_address = to_header
            
            # Extract domain
            domain_match = re.search(r'@(.+)$', email_address)
            if domain_match:
                return domain_match.group(1)
        
        return ''
    
    def is_suspicious_sender(self, email_data: Dict) -> bool:
        """Check if sender is suspicious"""
        sender_domain = self.extract_sender_domain(email_data)
        
        # Suspicious domain patterns
        suspicious_patterns = [
            r'\.tk$', r'\.ml$', r'\.ga$', r'\.cf$',  # Free domains
            r'\.xyz$', r'\.top$', r'\.club$', r'\.online$',  # New TLDs
            r'[0-9]{4,}',  # Many numbers
            r'[a-z]{20,}',  # Very long domain names
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, sender_domain, re.IGNORECASE):
                return True
        
        return False
    
    def get_email_summary(self, email_data: Dict) -> Dict:
        """Get a summary of email characteristics"""
        summary = {
            'sender_domain': self.extract_sender_domain(email_data),
            'recipient_domain': self.extract_recipient_domain(email_data),
            'subject_length': len(email_data['headers'].get('subject', '')),
            'body_length': len(email_data['body']['text']),
            'has_attachments': email_data['metadata']['has_attachments'],
            'attachment_count': email_data['metadata']['attachment_count'],
            'is_suspicious_sender': self.is_suspicious_sender(email_data),
            'spam_score': email_data['spam_indicators']['spam_score'],
            'priority': email_data['spam_indicators']['priority']
        }
        
        return summary 