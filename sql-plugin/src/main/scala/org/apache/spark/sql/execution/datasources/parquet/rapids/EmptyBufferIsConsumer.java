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

import com.google.common.collect.Iterators;
import org.apache.parquet.io.api.Binary;
import org.apache.spark.sql.execution.vectorized.rapids.WritableColumnVector;

public class EmptyBufferIsConsumer extends ByteBufferIsConsumer {

	public EmptyBufferIsConsumer() {
		super(Iterators.emptyIterator());
	}

	@Override
	public void readInts(int total, WritableColumnVector c, int rowId) {
		throw new AssertionError("We should NOT perform any reading from EmptyBufferIsConsumer");
	}

	@Override
	public void readLongs(int total, WritableColumnVector c, int rowId) {
		throw new AssertionError("We should NOT perform any reading from EmptyBufferIsConsumer");
	}

	@Override
	public void readFloats(int total, WritableColumnVector c, int rowId) {
		throw new AssertionError("We should NOT perform any reading from EmptyBufferIsConsumer");
	}

	@Override
	public void readDoubles(int total, WritableColumnVector c, int rowId) {
		throw new AssertionError("We should NOT perform any reading from EmptyBufferIsConsumer");
	}

	@Override
	public void readUIntsAsLongs(int total, WritableColumnVector c, int rowId) {
		throw new AssertionError("We should NOT perform any reading from EmptyBufferIsConsumer");
	}

	@Override
	public void readIntsAsShorts(int total, WritableColumnVector c, int rowId) {
		throw new AssertionError("We should NOT perform any reading from EmptyBufferIsConsumer");
	}

	@Override
	public void readIntsAsBytes(int total, WritableColumnVector c, int rowId) {
		throw new AssertionError("We should NOT perform any reading from EmptyBufferIsConsumer");
	}

	@Override
	public void readBinaries(int total, WritableColumnVector c, int rowId) {
		throw new AssertionError("We should NOT perform any reading from EmptyBufferIsConsumer");
	}

	@Override
	public byte getByte() {
		throw new AssertionError("We should NOT perform any reading from EmptyBufferIsConsumer");
	}

	@Override
	public int getInt() {
		throw new AssertionError("We should NOT perform any reading from EmptyBufferIsConsumer");}

	@Override
	public long getLong() {
		throw new AssertionError("We should NOT perform any reading from EmptyBufferIsConsumer");
	}

	@Override
	public float getFloat() {
		throw new AssertionError("We should NOT perform any reading from EmptyBufferIsConsumer");
	}

	@Override
	public double getDouble() {
		throw new AssertionError("We should NOT perform any reading from EmptyBufferIsConsumer");
	}

	@Override
	public Binary getBinary(int len) {
		throw new AssertionError("We should NOT perform any reading from EmptyBufferIsConsumer");
	}
}
