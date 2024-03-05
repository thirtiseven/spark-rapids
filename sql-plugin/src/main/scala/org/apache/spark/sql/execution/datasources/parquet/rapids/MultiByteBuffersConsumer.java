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

import org.apache.parquet.bytes.ByteBufferInputStream;
import org.apache.parquet.io.api.Binary;
import org.apache.spark.sql.execution.vectorized.rapids.HostWritableColumnVector;
import org.apache.spark.sql.execution.vectorized.rapids.WritableColumnVector;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Iterator;
import java.util.List;

public class MultiByteBuffersConsumer
{

	Iterator<ByteBuffer> iterator;
	ByteBuffer current = null;

	// Reference of underlying data structure of HeapByteBuffer, since we assume the BIS is
	// backed by the HeapByteBuffer
	byte[] hb;
	int arrayOffset;

	public MultiByteBuffersConsumer(ByteBufferInputStream bis) {
		List<ByteBuffer> buffers = bis.remainingBuffers();
		if (buffers.size() > 1) {
			System.err.printf("create a MultiByteBuffersConsumer with %d buffers\n", buffers.size());
		}
		iterator = buffers.iterator();
		if (iterator.hasNext()) {
			pointToNextBuffer();
		}
	}

	private void pointToNextBuffer() {
		current = iterator.next();
		current.order(ByteOrder.LITTLE_ENDIAN);
		assert current.hasArray();
		hb = current.array();
		arrayOffset = current.arrayOffset();
	}

	public void readInts(int total, WritableColumnVector c, int rowId) {
		int remaining = total * 4;
		int tgtOffset = rowId;

		while (remaining > 0) {
			if (!current.hasRemaining()) pointToNextBuffer();

			int size = Math.min(remaining, current.remaining());
			int srcOffset = this.arrayOffset + current.position();
			int sizeInRow = size >> 2;
			c.putIntsLittleEndian(tgtOffset, sizeInRow, hb, srcOffset);
			current.position(current.position() + size);
			tgtOffset += sizeInRow;
			remaining -= size;
		}
	}

	public void readLongs(int total, WritableColumnVector c, int rowId) {
		int remaining = total * 8;
		int tgtOffset = rowId;

		while (remaining > 0) {
			if (!current.hasRemaining()) pointToNextBuffer();

			int size = Math.min(remaining, current.remaining());
			int srcOffset = this.arrayOffset + current.position();
			int sizeInRow = size >> 3;
			c.putLongsLittleEndian(tgtOffset, sizeInRow, hb, srcOffset);
			current.position(current.position() + size);
			tgtOffset += sizeInRow;
			remaining -= size;
		}
	}

	public void readFloats(int total, WritableColumnVector c, int rowId) {
		int remaining = total * 4;
		int tgtOffset = rowId;

		while (remaining > 0) {
			if (!current.hasRemaining()) pointToNextBuffer();

			int size = Math.min(remaining, current.remaining());
			int srcOffset = this.arrayOffset + current.position();
			int sizeInRow = size >> 2;
			c.putFloatsLittleEndian(tgtOffset, sizeInRow, hb, srcOffset);
			current.position(current.position() + size);
			tgtOffset += sizeInRow;
			remaining -= size;
		}
	}

	public void readDoubles(int total, WritableColumnVector c, int rowId) {
		int remaining = total * 8;
		int tgtOffset = rowId;

		while (remaining > 0) {
			if (!current.hasRemaining()) pointToNextBuffer();

			int size = Math.min(remaining, current.remaining());
			int srcOffset = this.arrayOffset + current.position();
			int sizeInRow = size >> 3;
			c.putDoublesLittleEndian(tgtOffset, sizeInRow, hb, srcOffset);
			current.position(current.position() + size);
			tgtOffset += sizeInRow;
			remaining -= size;
		}
	}

	public void readUIntsAsLongs(int total, WritableColumnVector c, int rowId) {
		int remaining = total * 4;
		int tgtOffset = rowId;

		while (remaining > 0) {
			if (!current.hasRemaining()) pointToNextBuffer();

			int size = Math.min(remaining, current.remaining());
			for (int i = 0; i < size >> 2; ++i) {
				c.putLong(tgtOffset + i, Integer.toUnsignedLong(current.getInt()));
			}
			tgtOffset += size >> 3;
			remaining -= size;
		}
	}

	public void readIntsAsShorts(int total, WritableColumnVector c, int rowId) {
		int remaining = total * 4;
		int tgtOffset = rowId;

		while (remaining > 0) {
			if (!current.hasRemaining()) pointToNextBuffer();

			int size = Math.min(remaining, current.remaining());
			for (int i = 0; i < size >> 2; ++i) {
				c.putShort(tgtOffset + i, (short) current.getInt());
			}
			tgtOffset += size >> 1;
			remaining -= size;
		}
	}

	public void readIntsAsBytes(int total, WritableColumnVector c, int rowId) {
		int remaining = total * 4;
		int tgtOffset = rowId;

		while (remaining > 0) {
			if (!current.hasRemaining()) pointToNextBuffer();

			int size = Math.min(remaining, current.remaining());
			int pos = current.position();
			for (int i = 0; i < size >> 2; ++i) {
				c.putByte(tgtOffset + i, current.get(pos + (i << 2)));
			}
			tgtOffset += size;
			current.position(pos + size);
			remaining -= size;
		}
	}

	public void readBinaries(int total, WritableColumnVector c, int rowId) {
		HostWritableColumnVector vector = (HostWritableColumnVector) c;
		WritableColumnVector charVector = c.arrayData();

		for (int i = 0; i < total; ++i) {
			if (!current.hasRemaining()) pointToNextBuffer();

			int curLength = current.getInt();
			int prevOffset = charVector.getElementsAppended();

			if (curLength > 0) {
				charVector.reserve(prevOffset + curLength);

				int remainLen = curLength;
				int charOffset = prevOffset;
				while (remainLen > 0) {
					if (!current.hasRemaining()) pointToNextBuffer();

					int size = Math.min(remainLen, current.remaining());
					int bufPos = current.position();
					charVector.putBytes(charOffset, size, this.hb, this.arrayOffset + bufPos);
					charOffset += size;
					current.position(bufPos + size);
					remainLen -= size;
				}
				charVector.addElementsAppended(curLength);
			}

			vector.commitStringAppend(rowId + i, prevOffset, curLength);
		}
	}

	public byte getByte() {
		if (!current.hasRemaining()) pointToNextBuffer();
		return current.get();
	}

	public int getInt() {
		if (!current.hasRemaining()) pointToNextBuffer();
		return current.getInt();
	}

	public long getLong() {
		if (!current.hasRemaining()) pointToNextBuffer();
		return current.getLong();
	}

	public float getFloat() {
		if (!current.hasRemaining()) pointToNextBuffer();
		return current.getFloat();
	}

	public double getDouble() {
		if (!current.hasRemaining()) pointToNextBuffer();
		return current.getDouble();
	}

	public Binary getBinary(int len) {
		byte[] target = new byte[len];
		int targetOffset = 0;

		do {
			if (!current.hasRemaining()) pointToNextBuffer();

			int batchSize = Math.min(len - targetOffset, current.remaining());
			System.arraycopy(hb,this.arrayOffset + current.position(),
					target, targetOffset, batchSize);
			current.position(current.position() + batchSize);
			targetOffset += batchSize;
		}
		while (targetOffset < len);

		return Binary.fromConstantByteArray(target);
	}

	public void advance(long sizeInByte) {
		long remaining = sizeInByte;
		while (remaining > 0) {
			if (!current.hasRemaining()) pointToNextBuffer();

			int batchSize = (remaining >= current.remaining()) ? current.remaining() : (int) remaining;
			current.position(current.position() + batchSize);
			remaining -= batchSize;
		}
	}

}
