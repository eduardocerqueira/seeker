#date: 2025-07-29T16:47:41Z
#url: https://api.github.com/gists/b28ee969635c44b018f8f5f034840c94
#owner: https://api.github.com/users/rohithkarnati92

async def check_kafka_protocol(self, hostname: str, port: int = 9092) -> Dict:
    """Kafka protocol testing using aiokafka with proper cleanup"""
    start_time = time.time()
    producer = None

    try:
        # Create Kafka producer
        producer = AIOKafkaProducer(
            bootstrap_servers=f'{hostname}:{port}',
            request_timeout_ms=self.timeout * 1000,
            connections_max_idle_ms=self.timeout * 1000,
            max_batch_size=1,
            linger_ms=0,
            acks=1
        )

        # Start producer with timeout
        await asyncio.wait_for(
            producer.start(),
            timeout=self.timeout
        )
        
        # If we get here, producer started successfully
        return {
            'success': True,
            'protocol_verified': True,
            'service_info': f'Kafka broker accessible at {hostname}:{port}',
            'response_time': time.time() - start_time
        }

    except asyncio.TimeoutError:
        return {
            'success': False,
            'protocol_verified': False,
            'error': 'Kafka protocol timeout (likely DPI blocking)',
            'response_time': time.time() - start_time
        }
    except Exception as e:
        error_str = str(e).lower()

        if any(keyword in error_str for keyword in
               ['timeout', 'unable to connect', 'connection refused', 'no route to host']):
            return {
                'success': False,
                'protocol_verified': False,
                'error': f'Kafka protocol blocked: {str(e)}',
                'response_time': time.time() - start_time
            }
        elif any(keyword in error_str for keyword in ['authentication', 'authorization', 'sasl', 'security']):
            return {
                'success': True,
                'protocol_verified': True,
                'service_info': 'Kafka protocol accessible, auth/authorization issue (expected)',
                'response_time': time.time() - start_time,
                'auth_result': 'auth_required',
                'kafka_error': str(e)
            }
        else:
            return {
                'success': True,
                'protocol_verified': True,
                'service_info': f'Kafka protocol accessible, error: {e}',
                'response_time': time.time() - start_time,
                'kafka_error': str(e)
            }

    finally:
        # Improved cleanup logic
        if producer is not None:
            try:
                # Check if producer was started by checking internal state
                if hasattr(producer, '_closed') and not producer._closed:
                    # Producer was started, stop it properly
                    await asyncio.wait_for(producer.stop(), timeout=3)
                else:
                    # Producer was created but not started, close client connections
                    if hasattr(producer, '_client') and producer._client:
                        await producer._client.close()
            except asyncio.TimeoutError:
                logger.warning(f"Kafka producer stop timeout for {hostname}:{port}")
                # Force close if stop times out
                try:
                    if hasattr(producer, '_client') and producer._client:
                        await producer._client.close()
                except Exception as force_close_error:
                    logger.warning(f"Kafka producer force close error: {force_close_error}")
            except Exception as cleanup_error:
                logger.warning(f"Kafka producer cleanup error for {hostname}:{port}: {cleanup_error}")
                # Try force close as last resort
                try:
                    if hasattr(producer, '_client') and producer._client:
                        await producer._client.close()
                except:
                    pass  # Ignore force close errors