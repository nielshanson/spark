/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.mllib.recommendation

import breeze.linalg.{*, diag, inv, DenseMatrix, DenseVector, Matrix, Vector}
import scala.collection.mutable.ListBuffer

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.{IndexedRow, IndexedRowMatrix}
import org.apache.spark.rdd.RDD


case class ProductInfluence(productId : Int,
                        influenceScore: Double,
                        percentage: Double)

case class ProductExplanation(explainedPid: Int,
                        score: Double, productInfluence: Array[ProductInfluence])

case class UserExplanation(user: Int,
                           productExplanations: Array[ProductExplanation])



@SerialVersionUID(100L)
class ALSExplain extends Serializable {
  def explain( prodFactors: RDD[(Int, Array[Double])],
               ratings: RDD[Rating], lambda: Double, alpha: Double, topExplanation: Int):
  RDD[UserExplanation] = {
    val sc = prodFactors.context
    
    // Create row number for each product factor
    // ((prodId, prodFactor), prodRowNum)
    val indexedProdFactors = prodFactors.zipWithIndex()
    // (prodId, (prodRowNum, prodFactor))
    val productToIndex = indexedProdFactors.map( v => (v._1._1, (v._2, v._1._2)))
    // (prodRowNum, prodFactor)
    val indexedFactors = indexedProdFactors.map( v => (v._2, v._1._2))
    // Convert indexedFactors to Spark matricies and compute Y, YT, YTY
    val YIndexMatrix = toIndexedMatrix(indexedFactors)
    val Y = YIndexMatrix.toBlockMatrix()
    val YT = Y.transpose
    val YTY = YT.multiply(Y)
    
    // Create local versions of these to send to the executors
    val local_YT = sc.broadcast(YT.toLocalMatrix().asBreeze).value
    val local_Y = sc.broadcast(Y.toLocalMatrix().asBreeze).value
    val local_YTY = sc.broadcast(YTY.toLocalMatrix().asBreeze).value
    
    // Create maps for lookup
    val indexProdCollect = indexedProdFactors.map(v => (v._1._1, v._2)).collect()
    // (prodId to prodRow)
    val indexMap = sc.broadcast(indexProdCollect
      .map(keyValue => (keyValue._1, keyValue._2)).toMap).value
      // (prodRow to prodId)
    val productMap = sc.broadcast(indexProdCollect
      .map( keyValue => (keyValue._2, keyValue._1)).toMap).value
    
    // scalastyle:off
    println("indexedProdFactors:")
    indexedProdFactors.take(5).foreach(println)
    // ((0,[D@74087d81),0)
    // ((10,[D@5027cd31),1)
    // ((20,[D@6f606e87),2)
    // ((30,[D@30bb9392),3)
    // ((40,[D@7680471),4)
    println("productToIndex:")
    productToIndex.take(5).foreach(println)
    // productToIndex:
    // (0,(0,[D@56b1c4f6))
    // (10,(1,[D@73c2eab7))
    // (20,(2,[D@239d143))
    // (30,(3,[D@652c2be))
    // (40,(4,[D@5e1c20f6))
    println("indexedFactors:")
    indexedFactors.take(5).foreach(println)
    // (0,[D@25b6568a)
    // (1,[D@79d24c32)
    // (2,[D@556db970)
    // (3,[D@3a010811)
    // (4,[D@2d140bdd)
    // scalastyle:on
    
    val lambdaI = sc.broadcast(diag(DenseVector(List.fill(local_YTY.rows)
    (lambda).toArray))).value
    
    val userExplanationRDD = ratings
      .map( rating => (rating.product, rating))
      .join(productToIndex)
      .map{
        case (_, joinedRow) =>
          (joinedRow._1.user, (joinedRow._2, (joinedRow._1.rating * alpha)))
      }.groupByKey().map( group => UserExplanation(group._1,
      generateUserExplanation(group._2.toArray, local_YT, local_Y, local_YTY,
        lambdaI, indexMap, productMap, topExplanation)))
    userExplanationRDD
  }
  /** generate user Explanation for all products: [productId, productExplanation] */
  def generateUserExplanation(CuArray: Array[((Long, Array[Double]), Double)],
                              local_YT: Matrix[Double],
                              local_Y: Matrix[Double],
                              local_YTY: Matrix[Double], lambdaI: Matrix[Double],
                              indexLookup: Map[Int, Long],
                              productLookup: Map[Long, Int],
                              topExplanation: Int) : Array[ProductExplanation] = {
    val YTCUMinusI = CuArray.map{ case row =>
      (row._1._1, row._1._2.map(_ * row._2))}.sortBy(_._1).map(_._2)
    val YTCUMinusIMatrix = new DenseMatrix(local_YTY.rows,
      CuArray.length, YTCUMinusI.flatten.toArray)

    val YTCU = CuArray.map(row => (row._1._1, row._1._2.map( _ * (row._2 + 1))))
      .sortBy(_._1).map(_._2)
    val YTCUMatrix = new DenseMatrix(local_YTY.rows, CuArray.length, YTCU.flatten.toArray)

    val cuIndex = CuArray.sortBy(_._1._1).map(_._1._1)
      .zipWithIndex.map( row => (row._2, row._1)).toMap

    val Y = CuArray.map(row => (row._1._1, row._1._2)).sortBy(_._1).map(_._2)
    val YMatrix = new DenseMatrix(local_YTY.rows, CuArray.length, Y.flatten.toArray).t

    val A = YTCUMinusIMatrix * YMatrix
    val W = inv(local_YTY.toDenseMatrix + A.toDenseMatrix + lambdaI.toDenseMatrix).toDenseMatrix
    val S = (local_Y * W * YTCUMatrix).toDenseMatrix
    S(*, ::).map( x => generateProductExplanation(x, cuIndex, productLookup, topExplanation))
      .toArray.zipWithIndex.map(
      row => ProductExplanation(productLookup(row._2), row._1._1, row._1._2))
      .sortBy(_.explainedPid)
  }

  /** generate Product Explanation: productId, influenceScore, percentage */
  def generateProductExplanation(row: Vector[Double],
                                 cuIndex: Map[Int, Long],
                                 productLookup: Map[Long, Int],
                                 topExplanation: Int): (Double, Array[ProductInfluence]) =
  {
    val result : ListBuffer[ProductInfluence] = new ListBuffer[ProductInfluence]
    var sum: Double = 0.0
    for ( key <- cuIndex ) {
      val score = row(key._1)
      sum = sum + score
    }

    for ( key <- cuIndex ) {
      val score = row(key._1)
      val item = new ProductInfluence(
        productId = productLookup(key._2),
        influenceScore = score,
        percentage = score/sum
      )
      result += item
    }

    (sum, result.sortBy(-_.percentage).take(topExplanation).toArray)
  }

  /** Convert features Rdd to IndexedRowMatrix */
  def toIndexedMatrix(features: RDD[(Long, Array[Double])]): IndexedRowMatrix = {
    val rows = features.map { case (i, xs) => IndexedRow(i, Vectors.dense(xs)) }
    new IndexedRowMatrix(rows)
  }

}
