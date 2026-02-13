/*
 * Copyright (c) 2026, NVIDIA CORPORATION.
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

/*** spark-rapids-shim-json-lines
{"spark": "320"}
{"spark": "321"}
{"spark": "321cdh"}
{"spark": "322"}
{"spark": "323"}
{"spark": "324"}
{"spark": "330"}
{"spark": "330cdh"}
{"spark": "330db"}
{"spark": "331"}
{"spark": "332"}
{"spark": "332cdh"}
{"spark": "332db"}
{"spark": "333"}
{"spark": "334"}
{"spark": "340"}
{"spark": "341"}
{"spark": "341db"}
{"spark": "342"}
{"spark": "343"}
{"spark": "344"}
{"spark": "350"}
{"spark": "350db143"}
{"spark": "351"}
{"spark": "352"}
{"spark": "353"}
{"spark": "354"}
{"spark": "355"}
{"spark": "356"}
{"spark": "357"}
{"spark": "400"}
{"spark": "401"}
spark-rapids-shim-json-lines ***/
package com.nvidia.spark.rapids.shims

import scala.util.control.NonFatal

import com.nvidia.spark.rapids.{DataFromReplacementRule, RapidsConf, RapidsMeta, SparkPlanMeta, TargetSize}
import org.apache.hadoop.fs.Path
import org.apache.hadoop.mapred.{FileInputFormat => OldFileInputFormat}
import org.apache.hadoop.mapreduce.lib.input.{FileInputFormat => NewFileInputFormat}

import org.apache.spark.internal.Logging
import org.apache.spark.rdd.{HadoopRDD, NewHadoopRDD, RDD}
import org.apache.spark.sql.execution.{RDDScanExec, SparkPlan}
import org.apache.spark.sql.rapids.GpuSequenceFileRDDScanExec
import org.apache.spark.sql.types.BinaryType

class SequenceFileRDDScanExecMeta(
    plan: RDDScanExec,
    conf: RapidsConf,
    parent: Option[RapidsMeta[_, _, _]],
    rule: DataFromReplacementRule)
  extends SparkPlanMeta[RDDScanExec](plan, conf, parent, rule) with Logging {

  override def tagPlanForGpu(): Unit = {
    if (!conf.isSequenceFileRDDPhysicalReplaceEnabled) {
      willNotWorkOnGpu("SequenceFile RDD physical replacement is disabled")
      return
    }
    val fieldChecks = wrapped.output.forall { a =>
      val n = a.name
      val isKeyOrValue = n.equalsIgnoreCase("key") || n.equalsIgnoreCase("value")
      isKeyOrValue && a.dataType == BinaryType
    } && wrapped.output.nonEmpty
    if (!fieldChecks) {
      willNotWorkOnGpu("SequenceFile RDD replacement only supports BinaryType key/value output")
      return
    }
    if (!SequenceFileRDDScanExecMeta.isSimpleSequenceFileRDD(wrapped.inputRDD)) {
      willNotWorkOnGpu("RDD lineage is not a simple SequenceFile scan")
      return
    }
    if (SequenceFileRDDScanExecMeta.hasCompressedInput(
        wrapped.inputRDD, wrapped.inputRDD.context.hadoopConfiguration)) {
      willNotWorkOnGpu("Compressed SequenceFile input falls back to CPU RDDScan")
    }
  }

  override def convertToGpu(): com.nvidia.spark.rapids.GpuExec = {
    GpuSequenceFileRDDScanExec(wrapped, TargetSize(conf.gpuTargetBatchSizeBytes))
  }

  override def convertToCpu(): SparkPlan = wrapped
}

object SequenceFileRDDScanExecMeta extends Logging {
  private def isNewApiSequenceFileRDD(rdd: NewHadoopRDD[_, _]): Boolean = {
    try {
      val cls = classOf[NewHadoopRDD[_, _]]
      cls.getDeclaredFields
        .filter(_.getName.contains("inputFormatClass"))
        .exists { f =>
          f.setAccessible(true)
          val v = f.get(rdd)
          val c = v match {
            case c: Class[_] => c
            case other =>
              try {
                val vf = other.getClass.getDeclaredField("value")
                vf.setAccessible(true)
                vf.get(other).asInstanceOf[Class[_]]
              } catch {
                case _: Throwable => null
              }
          }
          c != null && c.getName.contains("SequenceFile")
        }
    } catch {
      case NonFatal(_) => false
    }
  }

  private def isOldApiSequenceFileRDD(rdd: HadoopRDD[_, _]): Boolean = {
    try {
      val m = rdd.getClass.getMethod("getJobConf")
      val jc = m.invoke(rdd).asInstanceOf[org.apache.hadoop.mapred.JobConf]
      val ifc = jc.get("mapred.input.format.class")
      ifc != null && ifc.contains("SequenceFile")
    } catch {
      case NonFatal(_) => false
    }
  }

  private def isSimpleSequenceFileRDD(
      rdd: RDD[_],
      seen: Set[Int] = Set.empty,
      nonSourceStages: Int = 0): Boolean = {
    val id = System.identityHashCode(rdd)
    if (seen.contains(id)) return false
    rdd match {
      case n: NewHadoopRDD[_, _] => isNewApiSequenceFileRDD(n)
      case h: HadoopRDD[_, _] => isOldApiSequenceFileRDD(h)
      case other =>
        if (other.dependencies.size != 1) {
          false
        } else {
          isSimpleSequenceFileRDD(other.dependencies.head.rdd, seen + id, nonSourceStages + 1)
        }
    }
  }

  private def collectInputPaths(rdd: RDD[_]): Seq[String] = {
    rdd match {
      case n: NewHadoopRDD[_, _] =>
        try {
          val cls = classOf[NewHadoopRDD[_, _]]
          cls.getDeclaredFields
            .filter(f => f.getName == "_conf" || f.getName.contains("_conf"))
            .flatMap { f =>
              f.setAccessible(true)
              val cv = f.get(n)
              val conf = cv match {
                case c: org.apache.hadoop.conf.Configuration => c
                case other =>
                  try {
                    val vf = other.getClass.getDeclaredField("value")
                    vf.setAccessible(true)
                    vf.get(other).asInstanceOf[org.apache.hadoop.conf.Configuration]
                  } catch {
                    case _: Throwable => null
                  }
              }
              val p = if (conf != null) conf.get(NewFileInputFormat.INPUT_DIR) else null
              Option(p).toSeq
            }.flatMap(_.split(",").map(_.trim)).filter(_.nonEmpty)
        } catch {
          case NonFatal(_) => Seq.empty
        }
      case h: HadoopRDD[_, _] =>
        try {
          val m = h.getClass.getMethod("getJobConf")
          val jc = m.invoke(h).asInstanceOf[org.apache.hadoop.mapred.JobConf]
          val paths = OldFileInputFormat.getInputPaths(jc)
          if (paths == null) Seq.empty else paths.map(_.toString).toSeq
        } catch {
          case NonFatal(_) => Seq.empty
        }
      case other if other.dependencies.size == 1 =>
        collectInputPaths(other.dependencies.head.rdd)
      case _ => Seq.empty
    }
  }

  private def findAnyFile(path: Path, conf: org.apache.hadoop.conf.Configuration): Option[Path] = {
    val fs = path.getFileSystem(conf)
    val statuses = fs.globStatus(path)
    if (statuses == null || statuses.isEmpty) {
      None
    } else {
      val first = statuses.head
      if (first.isFile) {
        Some(first.getPath)
      } else {
        val it = fs.listFiles(first.getPath, true)
        if (it.hasNext) Some(it.next().getPath) else None
      }
    }
  }

  private def isCompressedSequenceFile(
      file: Path,
      conf: org.apache.hadoop.conf.Configuration): Boolean = {
    var in: java.io.DataInputStream = null
    try {
      in = new java.io.DataInputStream(file.getFileSystem(conf).open(file))
      val magic = new Array[Byte](4)
      in.readFully(magic)
      if (!(magic(0) == 'S' && magic(1) == 'E' && magic(2) == 'Q')) {
        false
      } else {
        org.apache.hadoop.io.Text.readString(in)
        org.apache.hadoop.io.Text.readString(in)
        val isCompressed = in.readBoolean()
        val isBlockCompressed = in.readBoolean()
        isCompressed || isBlockCompressed
      }
    } catch {
      case NonFatal(e) =>
        logDebug(s"Failed probing sequencefile header $file: ${e.getMessage}")
        false
    } finally {
      if (in != null) in.close()
    }
  }

  def hasCompressedInput(rdd: RDD[_], hadoopConf: org.apache.hadoop.conf.Configuration): Boolean = {
    val paths = collectInputPaths(rdd)
    paths.exists { p =>
      try {
        findAnyFile(new Path(p), hadoopConf).exists(f => isCompressedSequenceFile(f, hadoopConf))
      } catch {
        case NonFatal(_) => false
      }
    }
  }
}

