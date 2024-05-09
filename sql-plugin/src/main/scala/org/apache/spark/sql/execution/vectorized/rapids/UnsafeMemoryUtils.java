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

package org.apache.spark.sql.execution.vectorized.rapids;

import java.lang.reflect.Field;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.nio.Buffer;
import java.nio.ByteBuffer;

public class UnsafeMemoryUtils {

	static Method copyDirectMemory;
	static Method freeDirectMemory;
	static Field getDirectAddress;

	static {
		try {
			Class<?> clz = Class.forName("ai.rapids.cudf.UnsafeMemoryAccessor");
			copyDirectMemory = clz.getMethod("copyMemory",
					Object.class, long.class, Object.class, long.class, long.class);
			copyDirectMemory.setAccessible(true);
			freeDirectMemory = clz.getMethod("free", long.class);
			freeDirectMemory.setAccessible(true);
		} catch (ClassNotFoundException | NoSuchMethodException e) {
			throw new RuntimeException(e);
		}

		try {
			getDirectAddress = Buffer.class.getDeclaredField("address");
			getDirectAddress.setAccessible(true);
		} catch (NoSuchFieldException e) {
			throw new RuntimeException(e);
		}
	}

	/**
	 * The reflection of `ai.rapids.cudf.UnsafeMemoryAccessor.copyMemory`
	 * Copy memory from one address to the other.
	 */
	public static void copyMemory(Object src, long srcOffset, Object dst, long dstOffset,
																long length) {
		try {
			copyDirectMemory.invoke(null,
					src,
					srcOffset,
					dst,
					dstOffset,
					length);
		} catch (IllegalAccessException | InvocationTargetException e) {
			throw new RuntimeException(e);
		}
	}

	/**
	 * The reflection of `ai.rapids.cudf.UnsafeMemoryAccessor.free`
	 */
	public static void freeMemory(long address) {
		try {
			freeDirectMemory.invoke(null, address);
		} catch (IllegalAccessException | InvocationTargetException e) {
			throw new RuntimeException(e);
		}
	}

	public static void freeDirectByteBuffer(ByteBuffer bb) {
		assert bb.isDirect();
		try {
			long address = (long) getDirectAddress.get(bb);
			freeMemory(address);
		} catch (IllegalAccessException e) {
			throw new RuntimeException(e);
		}
	}

}
