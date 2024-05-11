/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.sql.execution.datasources.parquet.rapids;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;

import com.github.luben.zstd.ZstdDirectBufferDecompressingStream;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.compress.CodecPool;
import org.apache.hadoop.io.compress.CompressionCodec;
import org.apache.hadoop.io.compress.Decompressor;
import org.apache.parquet.bytes.ByteBufferInputStream;
import org.apache.parquet.bytes.BytesInput;
import org.apache.parquet.hadoop.CodecFactory;
import org.apache.parquet.hadoop.codec.ZstandardCodec;
import org.apache.parquet.hadoop.metadata.CompressionCodecName;

public class ParquetDirectCodecFactory extends CodecFactory {

	public ParquetDirectCodecFactory(Configuration configuration, int pageSize) {
		super(configuration, pageSize);
	}

	@SuppressWarnings("deprecation")
	class DirectBytesDecompressor extends BytesDecompressor {

		private final CompressionCodec codec;
		private final Decompressor decompressor;

		DirectBytesDecompressor(CompressionCodecName codecName) {
			this.codec = getCodec(codecName);
			if (codec != null) {
				decompressor = CodecPool.getDecompressor(codec);
			} else {
				decompressor = null;
			}
		}

		@Override
		public BytesInput decompress(BytesInput bytes, int uncompressedSize) throws IOException {
			if (codec == null) {
				return bytes;
			}
			if (decompressor != null) {
				decompressor.reset();
			}
			if (!(codec instanceof ZstandardCodec)) {
				InputStream is = codec.createInputStream(bytes.toInputStream(), decompressor);
				return BytesInput.from(is, uncompressedSize);
			}

			return decompressZstd(bytes);
		}

		@Override
		public void decompress(
				ByteBuffer input, int compressedSize, ByteBuffer output, int uncompressedSize)
				throws IOException {
			ByteBuffer decompressed =
					decompress(BytesInput.from(input), uncompressedSize).toByteBuffer();
			output.put(decompressed);
		}

		@Override
		public void release() {
			if (decompressor != null) {
				CodecPool.returnDecompressor(decompressor);
			}
		}

		/**
		 * Perform zstd decompression with direct memory as input and output
		 * <p>
		 * 1. Exhaust the input data;
		 * 2. complete all the decompression work eagerly;
		 * 3. return uncompressed buffers;
		 */
		private BytesInput decompressZstd(BytesInput bytes) throws IOException {
			List<ByteBuffer> inputBuffers;
			try (ByteBufferInputStream is = bytes.toInputStream()) {
				inputBuffers = is.remainingBuffers();
			}
			if (inputBuffers.isEmpty()) {
				throw new IllegalArgumentException("Got empty ByteBufferInputStream");
			}
			// Compute the total size of compressed data
			int totalUncompressedSize = 0;
			ByteBuffer tmpBlk = null;

			List<ByteBuffer> decompressedBlocks = new ArrayList<>();
			ZstdDirectBufferDecompressingStream zis = null;
			try {
				for (ByteBuffer inputBuffer: inputBuffers) {
					zis = new ZstdDirectBufferDecompressingStream(inputBuffer);
					zis.setFinalize(false);
					while (zis.hasRemaining()) {
						if (tmpBlk == null || tmpBlk.remaining() < 64) {
							if (tmpBlk != null) {
								tmpBlk.flip();
								totalUncompressedSize += tmpBlk.remaining();
								decompressedBlocks.add(tmpBlk);
							}
							tmpBlk = ByteBuffer.allocateDirect(RECOMMENDED_BATCH_SIZE);
						}
						zis.read(tmpBlk);
					}
					zis.close();
					zis = null;
				}
				// append the tailing block if exists
				if (tmpBlk != null) {
					tmpBlk.flip();
					totalUncompressedSize += tmpBlk.remaining();
					decompressedBlocks.add(tmpBlk);
				}
				// merge all blocks into a continuous fused buffer
				ByteBuffer concatBuffer = ByteBuffer.allocate(totalUncompressedSize);
				for (ByteBuffer blk : decompressedBlocks) {
					concatBuffer.put(blk);
				}
				concatBuffer.flip();

				return BytesInput.from(concatBuffer);

			} catch (Throwable ex) {
				if (zis != null) {
					zis.close();
				}
				for (ByteBuffer buf: inputBuffers) {
					buf.clear();
				}
				for (ByteBuffer buf: decompressedBlocks) {
					buf.clear();
				}
				throw new RuntimeException(ex);
			}
		}
	}

	@Override
	@SuppressWarnings("deprecation")
	protected BytesDecompressor createDecompressor(CompressionCodecName codecName) {
		return new ParquetDirectCodecFactory.DirectBytesDecompressor(codecName);
	}

	private static final int RECOMMENDED_BATCH_SIZE = 128 * 1024;
}
